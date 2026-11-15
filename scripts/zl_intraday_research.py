"""
Zero Lag Trend bot on BTC intraday (M5 / M15 / M30, plus H1 for reference) — find improvements.

Engine follows ctrader-zerolag/ZeroLagTrendBot and exposes its parameters:
  entry mode (flip / pullback / both, pullback needs impulse), stop mode (band / ATR),
  reward:risk (0 = no target, exit on opposite flip), MTF filter, trading hours (UTC),
  loss filter (2 losses in a row on one side blocks that side), length / band multiplier.
Extra test-only filters: skip weekends, long only.

Method (to avoid curve fitting):
  In-sample (IS)      = first 3 years of the 5-year window   -> all choices are made here
  Out-of-sample (OOS) = last 2 years                         -> only used to check
  1) baseline = bot defaults
  2) every single change tested against baseline on IS
  3) greedy: keep adding the change that improves IS total R most (max 3), only if each
     step adds >= 5% and keeps >= 60 IS trades
  4) report baseline vs improved on IS and OOS, by timeframe

Data: data/BTCUSD/5m/YYYY.csv.gz (Binance BTCUSDT spot). Cost: % of price per round trip.
Usage: python scripts/zl_intraday_research.py [--cost-pct 0.06] [--years 5] [--out results/btc/intraday]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/BTCUSD/5m")
TFS = {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h"}

BASE = dict(length=70, mult=1.2, mode="flip", stop="band", atr_mult=1.5, rr=2.0,
            mtf=None, hours=None, no_weekend=False, long_only=False, loss_filter=True)

CHANGES = {
    "MTF: H1 trend agrees":        dict(mtf="H1"),
    "MTF: H4 trend agrees":        dict(mtf="H4"),
    "MTF: D1 trend agrees":        dict(mtf="D1"),
    "Hours 07-20 UTC":             dict(hours=(7, 20)),
    "Hours 12-21 UTC (NY)":        dict(hours=(12, 21)),
    "No weekends":                 dict(no_weekend=True),
    "R:R 1.5":                     dict(rr=1.5),
    "R:R 3":                       dict(rr=3.0),
    "No target (exit on flip)":    dict(rr=0.0),
    "Stop: ATR x1.5":              dict(stop="atr", atr_mult=1.5),
    "Stop: ATR x3":                dict(stop="atr", atr_mult=3.0),
    # loss filter must be off here: with pullback-only entries (or long only) a blocked side can
    # only be released by an opposite trade, which never comes, so the bot would stop trading
    "Entry: pullback":             dict(mode="pullback", loss_filter=False),
    "Entry: flip + pullback":      dict(mode="both"),
    "Length 35":                   dict(length=35),
    "Length 140":                  dict(length=140),
    "Length 280":                  dict(length=280),
    "Band x1.8":                   dict(mult=1.8),
    "Long only":                   dict(long_only=True, loss_filter=False),
    "Loss filter off":             dict(loss_filter=False),
}


# ------------------------------------------------------------------ data
def load_5m():
    parts = [pd.read_csv(p, parse_dates=["Datetime"]) for p in sorted(DATA.glob("*.csv.gz"))]
    df = pd.concat(parts).drop_duplicates("Datetime").sort_values("Datetime")
    df.columns = [c.lower() for c in df.columns]
    t = df.pop("datetime")
    df.index = t.dt.tz_localize("UTC") if t.dt.tz is None else t
    return df[["open", "high", "low", "close"]]


def resample(df, rule):
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


# ------------------------------------------------------------------ indicator
def zl_lines(H, L, C, length, mult):
    n, lag, w = len(C), (length - 1) // 2, length * 3
    zs = np.full(n, np.nan)
    zs[lag:] = C[lag:] + (C[lag:] - C[:-lag])
    ema_start = lag + length - 1
    alpha = 2 / (length + 1)
    zl = np.full(n, np.nan)
    zl[ema_start] = zs[lag:ema_start + 1].mean()
    # EMA recursion (vectorised via pandas ewm with the SMA seed)
    seeded = pd.Series(zs[ema_start:])
    seeded.iloc[0] = zl[ema_start]
    zl[ema_start:] = seeded.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    prev = np.r_[C[0], C[:-1]]
    tr = np.maximum(H - L, np.maximum(np.abs(H - prev), np.abs(L - prev)))
    tr[0] = H[0] - L[0]
    atr = np.full(n, np.nan)
    s = pd.Series(tr[length - 1:])
    s.iloc[0] = tr[:length].mean()
    atr[length - 1:] = s.ewm(alpha=1 / length, adjust=False).mean().to_numpy()   # Wilder RMA
    vol = pd.Series(atr).rolling(w).max().to_numpy() * mult
    start = max(ema_start, (length - 1) + (w - 1)) + 1
    up, dn = zl + vol, zl - vol
    cu = np.zeros(n, bool)
    cd = np.zeros(n, bool)
    cu[1:] = (C[1:] > up[1:]) & (C[:-1] <= up[:-1])
    cd[1:] = (C[1:] < dn[1:]) & (C[:-1] >= dn[:-1])
    ev = np.where(cd, -1, np.where(cu, 1, 0))          # bearish cross wins if both (Pine order)
    ev[:start] = 0
    trend = pd.Series(np.where(ev != 0, ev, np.nan)).ffill().fillna(0).to_numpy().astype(int)
    trend[:start] = 0
    return zl, vol, atr, trend, start


def valid_pullbacks(C, H, L, zl, atr, trend, start, impulse_atr=1.0):
    n = len(C)
    ok_up = np.zeros(n, bool)
    ok_dn = np.zeros(n, bool)
    d = 0
    ext = ref = 0.0
    imp = False
    for k in range(start + 1, n):
        if trend[k] != trend[k - 1]:
            d, ref = trend[k], C[k]
            ext = H[k] if d == 1 else L[k]
            imp = False
            continue
        if d == 1:
            ext = max(ext, H[k])
            if ext - ref >= impulse_atr * atr[k]:
                imp = True
            if C[k] > zl[k] and C[k - 1] <= zl[k - 1] and imp:
                ok_up[k], imp, ref = True, False, ext
        elif d == -1:
            ext = min(ext, L[k])
            if ref - ext >= impulse_atr * atr[k]:
                imp = True
            if C[k] < zl[k] and C[k - 1] >= zl[k - 1] and imp:
                ok_dn[k], imp, ref = True, False, ext
    return ok_up, ok_dn


# ------------------------------------------------------------------ backtest
class Ctx:
    """Per-timeframe arrays and caches."""

    def __init__(self, df, m5):
        self.df, self.m5 = df, m5
        self.O, self.H, self.L, self.C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
        self.idx = df.index
        self.hour = df.index.hour.to_numpy()
        self.wday = df.index.dayofweek.to_numpy()
        tr = np.maximum(self.H - self.L, np.maximum(np.abs(self.H - np.r_[self.C[0], self.C[:-1]]),
                                                    np.abs(self.L - np.r_[self.C[0], self.C[:-1]])))
        self.atr14 = pd.Series(tr).rolling(14).mean().to_numpy()     # cBot: ATR(14, Simple)
        self._lines, self._pb, self._mtf = {}, {}, {}

    def lines(self, length, mult):
        k = (length, mult)
        if k not in self._lines:
            self._lines[k] = zl_lines(self.H, self.L, self.C, length, mult)
        return self._lines[k]

    def pullbacks(self, length, mult):
        k = (length, mult)
        if k not in self._pb:
            zl, vol, atr, trend, start = self.lines(length, mult)
            self._pb[k] = valid_pullbacks(self.C, self.H, self.L, zl, atr, trend, start)
        return self._pb[k]

    def mtf(self, tf, length, mult):
        """Trend of the last CLOSED higher-timeframe bar, known at each bar's open."""
        k = (tf, length, mult)
        if k not in self._mtf:
            rule = {"H1": "1h", "H4": "4h", "D1": "1D"}[tf]
            h = resample(self.m5, rule)
            *_, trend, _ = zl_lines(h["high"].to_numpy(), h["low"].to_numpy(), h["close"].to_numpy(), length, mult)
            closes_at = h.index + pd.Timedelta(rule)
            s = pd.Series(trend, index=closes_at)
            self._mtf[k] = s.reindex(self.idx, method="ffill").fillna(0).to_numpy().astype(int)
        return self._mtf[k]


def run(ctx, p, cost_pct):
    O, H, L, C, idx = ctx.O, ctx.H, ctx.L, ctx.C, ctx.idx
    n = len(C)
    zl, vol, _, trend, start = ctx.lines(p["length"], p["mult"])
    if p["mode"] != "flip":
        ok_up, ok_dn = ctx.pullbacks(p["length"], p["mult"])
    mtf = ctx.mtf(p["mtf"], p["length"], p["mult"]) if p["mtf"] else None
    out = []
    pos = None
    streak_side, streak, blocked = 0, 0, 0

    def close(j, px, why):
        nonlocal pos, streak_side, streak, blocked
        side, e, s = pos["side"], pos["entry"], pos["stop"]
        cost = e * cost_pct / 100
        r = (side * (px - e) - cost) / (abs(e - s) + cost)
        out.append((pos["j"], j, side, e, s, px, why, r))
        if p["loss_filter"]:
            if r < 0:
                streak = streak + 1 if streak_side == side else 1
                streak_side = side
                if streak >= 2:
                    blocked = side
            else:
                streak_side, streak = 0, 0
        pos = None

    for j in range(start + 2, n):
        i = j - 1
        fu = trend[i] == 1 and trend[i - 1] <= 0
        fd = trend[i] == -1 and trend[i - 1] >= 0
        if pos is not None and ((fd and pos["side"] == 1) or (fu and pos["side"] == -1)):
            close(j, O[j], "flip")
        if pos is None:
            pu = pd_ = False
            if p["mode"] != "flip":
                pu = trend[i] == 1 and trend[i - 1] == 1 and ok_up[i]
                pd_ = trend[i] == -1 and trend[i - 1] == -1 and ok_dn[i]
            ls = (p["mode"] != "pullback" and fu) or (p["mode"] != "flip" and pu)
            ss = (p["mode"] != "pullback" and fd) or (p["mode"] != "flip" and pd_)
            if blocked == 1:
                ls = False
                if ss and not fd:
                    ss = False
            elif blocked == -1:
                ss = False
                if ls and not fu:
                    ls = False
            if p["long_only"]:
                ss = False
            side = 1 if ls else -1 if ss else 0
            if side and mtf is not None and mtf[j] != side:
                side = 0
            if side and p["hours"]:
                a, b = p["hours"]
                if not (a <= ctx.hour[j] < b):
                    side = 0
            if side and p["no_weekend"] and ctx.wday[j] >= 5:
                side = 0
            if side:
                e = O[j]
                if p["stop"] == "band":
                    s = zl[i] - vol[i] if side == 1 else zl[i] + vol[i]
                else:
                    s = e - side * ctx.atr14[i] * p["atr_mult"]
                if side * (e - s) > 0:
                    t = e + side * p["rr"] * abs(e - s) if p["rr"] > 0 else None
                    pos = {"j": j, "side": side, "entry": e, "stop": s, "tgt": t}
                    if blocked and blocked != side:
                        blocked, streak_side, streak = 0, 0, 0
        if pos is not None:
            sd, s, t = pos["side"], pos["stop"], pos["tgt"]
            o = O[j] if pos["j"] != j else pos["entry"]
            if (L[j] <= s) if sd == 1 else (H[j] >= s):
                close(j, min(o, s) if sd == 1 else max(o, s), "SL")
            elif t is not None and ((H[j] >= t) if sd == 1 else (L[j] <= t)):
                close(j, max(o, t) if sd == 1 else min(o, t), "TP")
    if pos is not None:
        close(n - 1, C[-1], "end")
    tr = pd.DataFrame(out, columns=["j_in", "j_out", "side", "entry", "stop", "exit", "reason", "R"])
    tr["entry_time"] = idx[tr["j_in"].to_numpy()]
    tr["exit_time"] = idx[tr["j_out"].to_numpy()]
    return tr


def stats(tr):
    if tr.empty:
        return dict(trades=0, win=np.nan, totR=0.0, avgR=np.nan, pf=np.nan, ddR=0.0, per_month=0.0)
    r = tr["R"].to_numpy()
    cum = np.cumsum(r)
    dd = (cum - np.maximum.accumulate(np.maximum(cum, 0))).min()
    months = max((tr["exit_time"].iloc[-1] - tr["entry_time"].iloc[0]).days / 30.44, 1)
    loss = -r[r < 0].sum()
    return dict(trades=len(r), win=round((r > 0).mean() * 100, 1), totR=round(r.sum(), 1),
                avgR=round(r.mean(), 3), pf=round(r[r > 0].sum() / loss, 2) if loss else np.inf,
                ddR=round(dd, 1), per_month=round(len(r) / months, 1))


def split(tr, cut):
    return tr[tr["entry_time"] < cut], tr[tr["entry_time"] >= cut]


def label(p):
    diffs = [k for k, ch in CHANGES.items() if all(p.get(a) == b for a, b in ch.items())
             and any(BASE[a] != b for a, b in ch.items())]
    return " + ".join(diffs) if diffs else "Bot defaults"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-pct", type=float, default=0.06)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--out", default="results/btc/intraday")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m5 = load_5m()
    end = m5.index[-1]
    t0 = end - pd.DateOffset(years=a.years)
    cut = end - pd.DateOffset(years=2)
    warm = t0 - pd.Timedelta(days=40)          # indicator warm-up before the test window
    m5 = m5[m5.index >= warm]
    print(f"5m bars {len(m5)}  window {t0:%Y-%m-%d} → {end:%Y-%m-%d}  OOS from {cut:%Y-%m-%d}", flush=True)

    rows, best_rows, final = [], [], {}
    for tf, rule in TFS.items():
        df = m5 if tf == "M5" else resample(m5, rule)
        ctx = Ctx(df, m5)
        cache = {}

        def evaluate(p):
            key = json.dumps(p, sort_keys=True, default=str)
            if key not in cache:
                tr = run(ctx, p, a.cost_pct)
                tr = tr[tr["entry_time"] >= t0]
                cache[key] = (tr, *split(tr, cut))
            return cache[key]

        base_tr, base_is, base_oos = evaluate(dict(BASE))
        b_is = stats(base_is)
        rows.append(dict(tf=tf, test="Bot defaults", **{f"IS {k}": v for k, v in b_is.items()},
                         **{f"OOS {k}": v for k, v in stats(base_oos).items()}))
        single = {}
        for name, ch in CHANGES.items():
            p = {**BASE, **ch}
            _, tis, toos = evaluate(p)
            s_is, s_oos = stats(tis), stats(toos)
            single[name] = s_is["totR"]
            rows.append(dict(tf=tf, test=name, **{f"IS {k}": v for k, v in s_is.items()},
                             **{f"OOS {k}": v for k, v in s_oos.items()}))
            print(f"{tf:4s} {name:28s} IS {s_is['totR']:8.1f}R n={s_is['trades']:5d} | OOS {s_oos['totR']:8.1f}R", flush=True)

        # greedy forward selection on IS only
        cur, cur_score, chosen = dict(BASE), b_is["totR"], []
        for _ in range(3):
            best = None
            for name, ch in CHANGES.items():
                if name in chosen or any(set(ch) & set(CHANGES[c]) for c in chosen):
                    continue
                p = {**cur, **ch}
                _, tis, _ = evaluate(p)
                s = stats(tis)
                if s["trades"] < 60:
                    continue
                if best is None or s["totR"] > best[1]:
                    best = (name, s["totR"], p)
            if best and best[1] > cur_score + max(5.0, abs(cur_score) * 0.05):
                chosen.append(best[0])
                cur, cur_score = best[2], best[1]
            else:
                break
        tr_all, tis, toos = evaluate(cur)
        final[tf] = dict(params=cur, chosen=chosen, trades=tr_all)
        best_rows.append(dict(tf=tf, version="Bot defaults", **{f"IS {k}": v for k, v in b_is.items()},
                              **{f"OOS {k}": v for k, v in stats(base_oos).items()}))
        best_rows.append(dict(tf=tf, version=" + ".join(chosen) if chosen else "(no change helped)",
                              **{f"IS {k}": v for k, v in stats(tis).items()},
                              **{f"OOS {k}": v for k, v in stats(toos).items()}))
        print(f"{tf}: chosen {chosen}", flush=True)

    # follow-up check of the one change that was positive in BOTH periods on M30 (H1 trend filter),
    # alone and with small additions. Note: picked after seeing OOS, so treat as a lead to forward-test.
    follow = []
    df30 = resample(m5, "30min")
    ctx30 = Ctx(df30, m5)
    for name, ch in [("Bot defaults", {}), ("MTF: H1 trend agrees", dict(mtf="H1")),
                     ("MTF H1 + R:R 3", dict(mtf="H1", rr=3.0)),
                     ("MTF H1 + no target", dict(mtf="H1", rr=0.0)),
                     ("MTF H1 + no weekends", dict(mtf="H1", no_weekend=True))]:
        for cost in (a.cost_pct, 0.10):
            tr = run(ctx30, {**BASE, **ch}, cost)
            tr = tr[tr["entry_time"] >= t0]
            tis, toos = split(tr, cut)
            yr = tr.groupby(tr["entry_time"].dt.year)["R"].sum().round(1)
            follow.append(dict(version=name, cost_pct=cost, **{f"IS {k}": v for k, v in stats(tis).items()},
                               **{f"OOS {k}": v for k, v in stats(toos).items()},
                               **{str(y): v for y, v in yr.items()}))
    follow = pd.DataFrame(follow)
    follow.to_csv(out / "m30_h1_filter_check.csv", index=False)

    grid = pd.DataFrame(rows)
    best = pd.DataFrame(best_rows)
    grid.to_csv(out / "all_tests.csv", index=False)
    best.to_csv(out / "baseline_vs_improved.csv", index=False)
    allt = []
    for tf, f in final.items():
        t = f["trades"].drop(columns=["j_in", "j_out"]).copy()
        t.insert(0, "tf", tf)
        allt.append(t)
    pd.concat(allt).to_csv(out / "improved_trades.csv", index=False)
    (out / "improved_params.json").write_text(json.dumps(
        {tf: {"changes": f["chosen"], "params": f["params"]} for tf, f in final.items()}, indent=2, default=str))

    cols = ["tf", "version", "IS trades", "IS win", "IS totR", "IS pf", "IS ddR",
            "OOS trades", "OOS win", "OOS totR", "OOS pf", "OOS ddR", "OOS per_month"]
    gcols = ["tf", "test", "IS trades", "IS totR", "IS pf", "OOS trades", "OOS totR", "OOS pf"]
    md = [f"# Zero Lag Trend on BTC intraday — improvements\n",
          f"Data: Binance BTCUSDT 5-min, {t0:%Y-%m-%d} → {end:%Y-%m-%d}. "
          f"In-sample (choices made here): to {cut:%Y-%m-%d}. Out-of-sample (check only): after. "
          f"Cost {a.cost_pct}% of price per round trip. Results in R.\n",
          "## Bot defaults vs improved version\n", best[cols].to_markdown(index=False),
          "\n\n## M30 + H1 trend filter (positive in both periods) — follow-up check\n",
          follow.drop(columns=[c for c in follow.columns if c.startswith(("IS ddR", "IS avgR", "OOS avgR", "IS per", "IS win"))]).to_markdown(index=False),
          "\n\n## Every single change (vs bot defaults)\n", grid[gcols].to_markdown(index=False)]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md[:4]))


if __name__ == "__main__":
    main()
