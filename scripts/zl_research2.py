"""
Deeper research on the Zero Lag M30 + H1-filter BTC setup.

Three periods:
  DESIGN   2021-11-14 → 2024-11-14  all choices are made here (Sharpe of daily returns)
  CHECK    2024-11-14 → 2026-11-14  already seen in the first study
  HOLDOUT  2018-01-01 → 2021-11-14  never looked at before this study, judged once at the end

New options on top of the bot (each one is a candidate change):
  exits   : break-even after +1R, ATR chandelier trailing stop, time stop
  regime  : efficiency-ratio (trendiness) minimum, band-width percentile filter
  volume  : signal candle volume above its 24h average
  trend   : D1 Zero Lag trend must not be against the trade
  plus a two-strategy portfolio: M30 setup + the D1 bot

Usage: python scripts/zl_research2.py [--cost-pct 0.06] [--out results/btc/research2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import zl_intraday_research as z

DESIGN = (pd.Timestamp("2021-11-14", tz="UTC"), pd.Timestamp("2024-11-14", tz="UTC"))
CHECK = (pd.Timestamp("2024-11-14", tz="UTC"), None)
HOLDOUT = (pd.Timestamp("2018-01-01", tz="UTC"), pd.Timestamp("2021-11-14", tz="UTC"))

BASE = {**z.BASE, "mtf": "H1", "d1_guard": False, "er_n": 48, "er_min": 0.0,
        "bw_lo": 0.0, "bw_hi": 1.0, "vol_k": 0.0, "be_at": 0.0, "trail_atr": 0.0, "time_stop": 0}

CHANGES = {
    "D1 trend not against":            dict(d1_guard=True),
    "Trendiness ER>=0.15":             dict(er_min=0.15),
    "Trendiness ER>=0.25":             dict(er_min=0.25),
    "Band width top 50%":              dict(bw_lo=0.5),
    "Band width bottom 50%":           dict(bw_hi=0.5),
    "Band width middle 20-80%":        dict(bw_lo=0.2, bw_hi=0.8),
    "Volume >= 1.0x avg":              dict(vol_k=1.0),
    "Volume >= 1.5x avg":              dict(vol_k=1.5),
    "Break-even at +1R":               dict(be_at=1.0),
    "Trail 3x ATR":                    dict(trail_atr=3.0),
    "Trail 4x ATR":                    dict(trail_atr=4.0),
    "Trail 5x ATR":                    dict(trail_atr=5.0),
    "Trail 6x ATR":                    dict(trail_atr=6.0),
    "Trail 5x ATR, no target":         dict(trail_atr=5.0, rr=0.0),
    "Trail 3x ATR, no target":         dict(trail_atr=3.0, rr=0.0),
    "Time stop 24 bars (12h)":         dict(time_stop=24),
    "Time stop 48 bars (1 day)":       dict(time_stop=48),
    "Time stop 72 bars (1.5 days)":    dict(time_stop=72),
    "Time stop 96 bars (2 days)":      dict(time_stop=96),
    "R:R 3":                           dict(rr=3.0),
    "No target (exit on flip)":        dict(rr=0.0),
    "No weekends":                     dict(no_weekend=True),
    "Band x1.5":                       dict(mult=1.5),
    "Length 50":                       dict(length=50),
    "Length 100":                      dict(length=100),
}


def load_5m_vol():
    parts = [pd.read_csv(p, parse_dates=["Datetime"]) for p in sorted(z.DATA.glob("*.csv.gz"))]
    df = pd.concat(parts).drop_duplicates("Datetime").sort_values("Datetime")
    df.columns = [c.lower() for c in df.columns]
    t = df.pop("datetime")
    df.index = t.dt.tz_localize("UTC") if t.dt.tz is None else t
    return df[["open", "high", "low", "close", "volume"]]


def resample(df, rule):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df:
        agg["volume"] = "sum"
    return df.resample(rule, label="left", closed="left").agg(agg).dropna()


class Ctx2(z.Ctx):
    def __init__(self, df, m5):
        super().__init__(df[["open", "high", "low", "close"]], m5[["open", "high", "low", "close"]])
        self.V = df["volume"].to_numpy()
        self.vavg = pd.Series(self.V).rolling(48).mean().to_numpy()
        self._er, self._bw = {}, {}

    def er(self, n):
        if n not in self._er:
            c = pd.Series(self.C)
            self._er[n] = ((c - c.shift(n)).abs() / c.diff().abs().rolling(n).sum()).to_numpy()
        return self._er[n]

    def bw_pct(self, length, mult):
        """Band width / price, as percentile vs the previous 30 days of bars."""
        k = (length, mult)
        if k not in self._bw:
            _, vol, *_ = self.lines(length, mult)
            w = pd.Series(vol / self.C)
            self._bw[k] = w.rolling(1440, min_periods=500).rank(pct=True).to_numpy()
        return self._bw[k]


def run2(ctx, p, cost_pct):
    O, H, L, C, idx = ctx.O, ctx.H, ctx.L, ctx.C, ctx.idx
    n = len(C)
    zl, vol, _, trend, start = ctx.lines(p["length"], p["mult"])
    mtf = ctx.mtf(p["mtf"], p["length"], p["mult"]) if p["mtf"] else None
    d1 = ctx.mtf("D1", p["length"], p["mult"]) if p["d1_guard"] else None
    er = ctx.er(p["er_n"]) if p["er_min"] > 0 else None
    bw = ctx.bw_pct(p["length"], p["mult"]) if (p["bw_lo"] > 0 or p["bw_hi"] < 1) else None
    atr = ctx.atr14
    out, pos = [], None
    streak_side = streak = blocked = 0

    def close(j, px, why):
        nonlocal pos, streak_side, streak, blocked
        side, e, s0 = pos["side"], pos["entry"], pos["stop0"]
        cost = e * cost_pct / 100
        r = (side * (px - e) - cost) / (abs(e - s0) + cost)
        out.append((pos["j"], j, side, e, s0, px, why, r))
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
        if pos is not None:
            # stop management decided on the last closed bar, applied from this bar
            sd, e, risk = pos["side"], pos["entry"], abs(pos["entry"] - pos["stop0"])
            pos["best"] = max(pos["best"], H[i]) if sd == 1 else min(pos["best"], L[i])
            if p["be_at"] > 0 and sd * (pos["best"] - e) >= p["be_at"] * risk:
                be = e + sd * e * cost_pct / 100
                pos["stop"] = max(pos["stop"], be) if sd == 1 else min(pos["stop"], be)
            if p["trail_atr"] > 0:
                tr_ = pos["best"] - sd * p["trail_atr"] * atr[i]
                pos["stop"] = max(pos["stop"], tr_) if sd == 1 else min(pos["stop"], tr_)
            if p["time_stop"] and (j - pos["j"]) >= p["time_stop"]:
                close(j, O[j], "time")
        if pos is not None and ((fd and pos["side"] == 1) or (fu and pos["side"] == -1)):
            close(j, O[j], "flip")
        if pos is None:
            ls, ss = fu, fd
            if blocked == 1:
                ls = False
            elif blocked == -1:
                ss = False
            if p["long_only"]:
                ss = False
            side = 1 if ls else -1 if ss else 0
            if side and mtf is not None and mtf[j] != side:
                side = 0
            if side and d1 is not None and d1[j] == -side:
                side = 0
            if side and er is not None and not (er[i] >= p["er_min"]):
                side = 0
            if side and bw is not None and not (p["bw_lo"] <= bw[i] <= p["bw_hi"]):
                side = 0
            if side and p["vol_k"] > 0 and not (ctx.V[i] >= p["vol_k"] * ctx.vavg[i]):
                side = 0
            if side and p["hours"]:
                a, b = p["hours"]
                if not (a <= ctx.hour[j] < b):
                    side = 0
            if side and p["no_weekend"] and ctx.wday[j] >= 5:
                side = 0
            if side:
                e = O[j]
                s = zl[i] - vol[i] if side == 1 else zl[i] + vol[i]
                if p["stop"] == "atr":
                    s = e - side * atr[i] * p["atr_mult"]
                if side * (e - s) > 0:
                    t = e + side * p["rr"] * abs(e - s) if p["rr"] > 0 else None
                    pos = {"j": j, "side": side, "entry": e, "stop": s, "stop0": s, "tgt": t,
                           "best": e}
                    if blocked and blocked != side:
                        blocked, streak_side, streak = 0, 0, 0
        if pos is not None:
            sd, s, t = pos["side"], pos["stop"], pos["tgt"]
            o = O[j] if pos["j"] != j else pos["entry"]
            if (L[j] <= s) if sd == 1 else (H[j] >= s):
                close(j, min(o, s) if sd == 1 else max(o, s), "SL" if s == pos["stop0"] else "trail")
            elif t is not None and ((H[j] >= t) if sd == 1 else (L[j] <= t)):
                close(j, max(o, t) if sd == 1 else min(o, t), "TP")
    if pos is not None:
        close(n - 1, C[-1], "end")
    tr = pd.DataFrame(out, columns=["j_in", "j_out", "side", "entry", "stop", "exit", "reason", "R"])
    tr["entry_time"] = idx[tr["j_in"].to_numpy()]
    tr["exit_time"] = idx[tr["j_out"].to_numpy()]
    return tr


# ------------------------------------------------------------------ metrics
def period(tr, per):
    a, b = per
    m = tr["entry_time"] >= a
    if b is not None:
        m &= tr["entry_time"] < b
    return tr[m]


def daily(tr, per, end):
    a, b = per
    days = pd.date_range(a.normalize(), (b or end).normalize(), freq="D", tz="UTC", inclusive="left")
    return tr.set_index("exit_time")["R"].resample("D").sum().reindex(days, fill_value=0.0)


def metrics(tr, per, end, risk=1.0):
    t = period(tr, per)
    d = daily(t, per, end) * risk / 100
    if t.empty or d.std() == 0:
        return dict(trades=0, sharpe=np.nan, totR=0.0, pf=np.nan, win=np.nan, ddR=0.0, cagr=np.nan, maxdd=np.nan)
    r = t["R"].to_numpy()
    eq = (1 + d).cumprod()
    yrs = len(d) / 365
    cum = np.cumsum(r)
    loss = -r[r < 0].sum()
    return dict(trades=len(r), sharpe=round(d.mean() / d.std() * np.sqrt(365), 2), totR=round(r.sum(), 1),
                pf=round(r[r > 0].sum() / loss, 2) if loss else np.inf, win=round((r > 0).mean() * 100, 1),
                ddR=round((cum - np.maximum.accumulate(np.maximum(cum, 0))).min(), 1),
                cagr=round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
                maxdd=round(((eq / eq.cummax()) - 1).min() * 100, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-pct", type=float, default=0.06)
    ap.add_argument("--out", default="results/btc/research2")
    ap.add_argument("--holdout", action="store_true", help="also score the 2018-2021 holdout")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m5 = load_5m_vol()
    end = m5.index[-1]
    first = m5.index[0]
    print(f"5m data {first} → {end}", flush=True)
    m30 = resample(m5, "30min")
    ctx = Ctx2(m30, m5)
    pers = {"DESIGN": DESIGN, "CHECK": CHECK}
    if a.holdout:
        pers["HOLDOUT"] = HOLDOUT

    cache = {}

    def ev(p):
        k = json.dumps(p, sort_keys=True, default=str)
        if k not in cache:
            cache[k] = run2(ctx, p, a.cost_pct)
        return cache[k]

    def row(name, p):
        tr = ev(p)
        r = {"test": name}
        for pn, per in pers.items():
            for k, v in metrics(tr, per, end).items():
                r[f"{pn} {k}"] = v
        return r

    rows = [row("Bot defaults (no H1 filter)", {**BASE, "mtf": None}), row("M30 + H1 filter (start point)", BASE)]
    base_sh = rows[1]["DESIGN sharpe"]
    for name, ch in CHANGES.items():
        rows.append(row(name, {**BASE, **ch}))
        print(f"{name:30s} " + "  ".join(f"{pn} sh {rows[-1][pn + ' sharpe']} R {rows[-1][pn + ' totR']} n {rows[-1][pn + ' trades']}"
                                         for pn in pers), flush=True)

    # Selection rule (fixed before looking at HOLDOUT):
    #  1) naive greedy on DESIGN Sharpe alone, shown for comparison (it over-fits)
    #  2) robust: keep only changes that raise Sharpe in BOTH DESIGN and CHECK, try every
    #     non-conflicting combination of them, pick the best worst-case (min of the two) Sharpe
    import itertools
    cur, cur_sh, greedy = dict(BASE), base_sh, []
    for _ in range(3):
        best = None
        for name, ch in CHANGES.items():
            if name in greedy or any(set(ch) & set(CHANGES[c]) for c in greedy):
                continue
            p = {**cur, **ch}
            m = metrics(ev(p), DESIGN, end)
            if m["trades"] >= 150 and (best is None or m["sharpe"] > best[1]):
                best = (name, m["sharpe"], p)
        if best and best[1] >= cur_sh + 0.10:
            greedy.append(best[0])
            cur, cur_sh = best[2], best[1]
        else:
            break
    greedy_row = row("Naive greedy on DESIGN: " + " + ".join(greedy), cur)

    base_ck = rows[1]["CHECK sharpe"]
    good = [r["test"] for r in rows[2:] if r["DESIGN sharpe"] > base_sh and r["CHECK sharpe"] > base_ck
            and r["DESIGN trades"] >= 150]
    best = (min(base_sh, base_ck), [], dict(BASE))
    combos = []
    for k in range(1, 4):
        for combo in itertools.combinations(good, k):
            keys = [set(CHANGES[c]) for c in combo]
            if any(a & b for a, b in itertools.combinations(keys, 2)):
                continue
            p = {**BASE}
            for c in combo:
                p.update(CHANGES[c])
            md_, mc_ = metrics(ev(p), DESIGN, end), metrics(ev(p), CHECK, end)
            combos.append({"combo": " + ".join(combo), "DESIGN sharpe": md_["sharpe"], "CHECK sharpe": mc_["sharpe"],
                           "worst": min(md_["sharpe"], mc_["sharpe"]), "DESIGN trades": md_["trades"]})
            if min(md_["sharpe"], mc_["sharpe"]) > best[0]:
                best = (min(md_["sharpe"], mc_["sharpe"]), list(combo), p)
    # prefer the simplest combo whose worst-case Sharpe is within 0.05 of the best
    for c in sorted(combos, key=lambda c: (c["combo"].count(" + "), -c["worst"])):
        if c["worst"] >= best[0] - 0.05:
            names = c["combo"].split(" + ")
            p = {**BASE}
            for nm in names:
                p.update(CHANGES[nm])
            best = (c["worst"], names, p)
            break
    chosen, cur = best[1], best[2]
    pd.DataFrame(combos).sort_values("worst", ascending=False).to_csv(out / "robust_combos.csv", index=False)
    improved = row(" + ".join(chosen) if chosen else "(no robust improvement)", cur)
    print("passed both periods:", good, flush=True)
    print("chosen:", chosen, flush=True)

    # portfolio with the D1 bot (default settings), 1% risk each
    d1 = resample(m5, "1D")
    ctxd = Ctx2(d1, m5)
    d1_tr = run2(ctxd, {**BASE, "mtf": None}, a.cost_pct)
    port = []
    for label, tr in (("D1 bot alone", d1_tr), ("M30+H1 alone", ev(BASE)), ("Improved M30 alone", ev(cur))):
        r = {"test": label}
        for pn, per in pers.items():
            for k, v in metrics(tr, per, end).items():
                r[f"{pn} {k}"] = v
        port.append(r)
    for label, tr in (("Portfolio: M30+H1 + D1 bot", ev(BASE)), ("Portfolio: improved M30 + D1 bot", ev(cur))):
        r = {"test": label}
        for pn, per in pers.items():
            dd = (daily(period(tr, per), per, end) + daily(period(d1_tr, per), per, end)) / 100
            eq = (1 + dd).cumprod()
            yrs = len(dd) / 365
            r[f"{pn} sharpe"] = round(dd.mean() / dd.std() * np.sqrt(365), 2)
            r[f"{pn} cagr"] = round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 1)
            r[f"{pn} maxdd"] = round(((eq / eq.cummax()) - 1).min() * 100, 1)
            r[f"{pn} trades"] = len(period(tr, per)) + len(period(d1_tr, per))
            r[f"{pn} corr"] = round(daily(period(tr, per), per, end).corr(daily(period(d1_tr, per), per, end)), 2)
        port.append(r)

    grid = pd.DataFrame(rows)
    grid.to_csv(out / "all_changes.csv", index=False)
    pd.DataFrame([rows[1], improved, greedy_row]).to_csv(out / "start_vs_improved.csv", index=False)
    pd.DataFrame(port).to_csv(out / "portfolio.csv", index=False)
    ev(cur).drop(columns=["j_in", "j_out"]).to_csv(out / "improved_trades.csv", index=False)
    ev(BASE).drop(columns=["j_in", "j_out"]).to_csv(out / "start_trades.csv", index=False)
    d1_tr.drop(columns=["j_in", "j_out"]).to_csv(out / "d1_trades.csv", index=False)
    (out / "improved_params.json").write_text(json.dumps({"changes": chosen, "params": cur}, indent=2, default=str))

    keys = ["trades", "sharpe", "totR", "pf", "cagr", "maxdd"]
    cols = ["test"] + [f"{pn} {k}" for pn in pers for k in keys]
    pcols = ["test"] + [f"{pn} {k}" for pn in pers for k in ("trades", "sharpe", "cagr", "maxdd")]
    md = ["# Zero Lag M30 + H1 on BTC — deeper research\n",
          f"Periods: DESIGN {DESIGN[0]:%Y-%m-%d}→{DESIGN[1]:%Y-%m-%d} (choices made here), "
          f"CHECK {CHECK[0]:%Y-%m-%d}→{end:%Y-%m-%d}"
          + (f", HOLDOUT {HOLDOUT[0]:%Y-%m-%d}→{HOLDOUT[1]:%Y-%m-%d} (untouched until the end)" if a.holdout else "")
          + f". Cost {a.cost_pct}%. Sharpe from daily returns at 1% risk/trade, annualised (365). "
            "CAGR / max DD at 1% risk.\n",
          "## Start point vs improved (robust pick) vs naive greedy\n",
          pd.DataFrame([rows[1], improved, greedy_row])[cols].to_markdown(index=False),
          "\n\n## Portfolio with the D1 bot (1% risk each)\n", pd.DataFrame(port).reindex(columns=pcols).to_markdown(index=False),
          "\n\n## Every change tested (one at a time vs start point)\n", grid[cols].to_markdown(index=False)]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md[:5]))


if __name__ == "__main__":
    main()
