"""
BTC/USD side-by-side backtest: Zero Lag Trend, SMA 9/18 pullback, CRT.

Rules are ported from GR's own code:
  Zero Lag  -> ctrader-zerolag/ZeroLagTrendBot (TrendFlip entry, Band stop, 2R TP,
               close on opposite flip, 2-loss side block)
  SMA 9/18  -> sma-signals/worker.js engine() (10-candle impulse, SMA 9 touch,
               60% body bounce candle, stop order, swing stop, 1:2, 1h SMA50 trend,
               max 1 cross in last 50 candles). London/NY session filter OFF (BTC is 24/7).
  CRT       -> eurusd-crt-backtest (sweep of previous HTF candle + close back inside,
               entry next LTF open, stop beyond sweep wick, target = opposite side)

Data: data/BTCUSD/1h.csv (~2 years) and data/BTCUSD/daily.csv (2014 onward), Yahoo BTC-USD.
Costs: COST_PCT of entry price per round trip (spread + slippage), charged on every trade.
All results in R (1R = the trade's stop distance + cost). Same-bar SL+TP counts as SL.

Usage: python scripts/btc_backtest.py [--cost-pct 0.06] [--out results/btc]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/BTCUSD")


# ----------------------------------------------------------------- data
def load(name):
    df = pd.read_csv(DATA / name, parse_dates=["Datetime"])
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"datetime": "t"})
    if df["t"].dt.tz is None:
        df["t"] = df["t"].dt.tz_localize("UTC")
    return df.drop_duplicates("t").sort_values("t").set_index("t")[["open", "high", "low", "close"]]


def resample(df, rule):
    return df.resample(rule, label="left", closed="left", origin="epoch").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


# ----------------------------------------------------------------- trade helpers
def r_mult(side, entry, stop, exit_, cost_pct):
    cost = entry * cost_pct / 100
    return (side * (exit_ - entry) - cost) / (abs(entry - stop) + cost)


def intrabar_exit(side, o, h, l, stop, tgt):
    """SL first if both touched; gaps fill at the open."""
    if side == 1:
        if l <= stop:
            return min(o, stop), "SL"
        if h >= tgt:
            return max(o, tgt), "TP"
    else:
        if h >= stop:
            return max(o, stop), "SL"
        if l <= tgt:
            return min(o, tgt), "TP"
    return None, None


# ----------------------------------------------------------------- Zero Lag
def zero_lag(df, cost_pct, length=70, mult=1.2, rr=2.0, losses_to_block=2):
    O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    n, lag, w = len(C), (length - 1) // 2, length * 3
    zs = np.full(n, np.nan)
    zs[lag:] = C[lag:] + (C[lag:] - C[:-lag])
    zl = np.full(n, np.nan)
    ema_start = lag + length - 1
    zl[ema_start] = zs[lag:ema_start + 1].mean()
    a = 2 / (length + 1)
    for i in range(ema_start + 1, n):
        zl[i] = a * zs[i] + (1 - a) * zl[i - 1]
    tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
    tr[0] = H[0] - L[0]
    atr = np.full(n, np.nan)
    atr[length - 1] = tr[:length].mean()
    for i in range(length, n):
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length
    vol = pd.Series(atr).rolling(w).max().to_numpy() * mult
    start = max(ema_start, (length - 1) + (w - 1)) + 1
    trend = np.zeros(n, dtype=int)
    t = 0
    for i in range(start, n):
        if C[i] > zl[i] + vol[i] and C[i - 1] <= zl[i - 1] + vol[i - 1]:
            t = 1
        if C[i] < zl[i] - vol[i] and C[i - 1] >= zl[i - 1] - vol[i - 1]:
            t = -1
        trend[i] = t

    trades, pos = [], None
    streak_side, streak, blocked = None, 0, None

    def close(j, price, why):
        nonlocal pos, streak_side, streak, blocked
        r = r_mult(pos["side"], pos["entry"], pos["stop"], price, cost_pct)
        trades.append({**pos, "exit_time": df.index[j], "exit": price, "reason": why, "R": r})
        if r < 0:
            if streak_side == pos["side"]:
                streak += 1
            else:
                streak_side, streak = pos["side"], 1
            if streak >= losses_to_block:
                blocked = pos["side"]
        else:
            streak_side, streak = None, 0
        pos = None

    for j in range(start + 2, n):
        i = j - 1                                   # last closed bar; act at open of j
        flip_up = trend[i] == 1 and trend[i - 1] <= 0
        flip_dn = trend[i] == -1 and trend[i - 1] >= 0
        if pos and ((flip_dn and pos["side"] == 1) or (flip_up and pos["side"] == -1)):
            close(j, O[j], "flip")
        side = 1 if flip_up else -1 if flip_dn else 0
        if side and blocked == side:
            side = 0
        if side and pos is None:
            entry = O[j]
            stop = zl[i] - vol[i] if side == 1 else zl[i] + vol[i]
            if side * (entry - stop) > 0:
                pos = {"side": side, "entry_time": df.index[j], "entry": entry, "stop": stop,
                       "tgt": entry + side * rr * abs(entry - stop)}
                if blocked is not None and blocked != side:
                    blocked, streak_side, streak = None, None, 0
        if pos:
            px, why = intrabar_exit(pos["side"], O[j], H[j], L[j], pos["stop"], pos["tgt"])
            if px is not None:
                close(j, px, why)
    if pos:
        close(n - 1, C[-1], "end")
    return trades


# ----------------------------------------------------------------- SMA 9/18 pullback
def sma_pullback(df, cost_pct, N=10, rr=2.0, c_max=1, c_look=50, buffer_pct=0.02):
    O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    n = len(C)
    s9 = pd.Series(C).rolling(9).mean().to_numpy()
    s18 = pd.Series(C).rolling(18).mean().to_numpy()
    R = pd.Series(H - L).rolling(N).sum().to_numpy()
    s50 = pd.Series(C).rolling(50).mean().to_numpy()
    trend = np.zeros(n, dtype=int)                 # sign(prev close - SMA50 at prev bar)
    trend[1:] = np.nan_to_num(np.sign(C[:-1] - s50[:-1])).astype(int)
    cross = np.zeros(n, dtype=int)
    for i in range(1, n):
        x = (s9[i - 1] <= s18[i - 1] and s9[i] > s18[i]) or (s9[i - 1] >= s18[i - 1] and s9[i] < s18[i])
        cross[i] = cross[i - 1] + int(x)
    chop_ok = lambda i: cross[i] - cross[max(0, i - c_look)] - 1 <= c_max

    trades = []
    s_dir = st = 0
    ref = need = ext = sw = 0.0
    p_entry = p_stop = None
    pos = None

    def exit_check(i):
        nonlocal pos
        px, why = intrabar_exit(pos["side"], O[i] if pos["entry_time"] != df.index[i] else pos["entry"],
                                H[i], L[i], pos["stop"], pos["tgt"])
        if px is not None:
            trades.append({**pos, "exit_time": df.index[i], "exit": px, "reason": why,
                           "R": r_mult(pos["side"], pos["entry"], pos["stop"], px, cost_pct)})
            pos = None

    for i in range(1, n):
        if pos:
            exit_check(i)
        if pos:
            continue
        up = i >= 19 and s9[i - 1] <= s18[i - 1] and s9[i] > s18[i]
        dn = i >= 19 and s9[i - 1] >= s18[i - 1] and s9[i] < s18[i]
        if (up and s_dir == 1) or (dn and s_dir == -1):
            up = dn = False
        if up or dn:
            d = 1 if up else -1
            p_entry = p_stop = None
            if trend[i] == d and chop_ok(i):
                s_dir, st, ref, need, ext = d, 1, C[i], R[i], C[i]
            else:
                s_dir = st = 0
            continue
        if not s_dir:
            continue
        if p_entry is not None:
            trig = H[i] > p_entry if s_dir == 1 else L[i] < p_entry
            if trig:
                fill = max(O[i], p_entry) if s_dir == 1 else min(O[i], p_entry)
                pos = {"side": s_dir, "entry_time": df.index[i], "entry": fill, "stop": p_stop,
                       "tgt": fill + s_dir * rr * abs(fill - p_stop)}
                s_dir = st = 0
                p_entry = p_stop = None
                exit_check(i)
                continue
        if st == 1:
            ext = max(ext, H[i]) if s_dir == 1 else min(ext, L[i])
            if s_dir * (ext - ref) >= need:
                st = 2
        elif st == 2:
            if (L[i] <= s9[i]) if s_dir == 1 else (H[i] >= s9[i]):
                st, sw = 3, (L[i] if s_dir == 1 else H[i])
        if st == 3:
            sw = min(sw, L[i]) if s_dir == 1 else max(sw, H[i])
            rg, body = H[i] - L[i], s_dir * (C[i] - O[i])
            if rg > 0 and body >= 0.6 * rg and s_dir * (C[i] - s9[i]) > 0:
                p_entry = H[i] if s_dir == 1 else L[i]
                p_stop = sw - s_dir * C[i] * buffer_pct / 100
    if pos:
        trades.append({**pos, "exit_time": df.index[-1], "exit": C[-1], "reason": "end",
                       "R": r_mult(pos["side"], pos["entry"], pos["stop"], C[-1], cost_pct)})
    return trades


# ----------------------------------------------------------------- CRT
def crt(ltf, htf, cost_pct, buffer_pct=0.02):
    rng = htf[["high", "low"]].shift(1).rename(columns={"high": "rh", "low": "rl"})
    w = ltf.join(rng.reindex(ltf.index, method="ffill")).dropna()
    O, H, L, C = (w[c].to_numpy() for c in ("open", "high", "low", "close"))
    RH, RL = w["rh"].to_numpy(), w["rl"].to_numpy()
    trades, pos = [], None
    for i in range(len(w) - 1):
        if pos:
            px, why = intrabar_exit(pos["side"], O[i], H[i], L[i], pos["stop"], pos["tgt"])
            if px is not None:
                trades.append({**pos, "exit_time": w.index[i], "exit": px, "reason": why,
                               "R": r_mult(pos["side"], pos["entry"], pos["stop"], px, cost_pct)})
                pos = None
        if pos is None:
            side = 0
            if L[i] < RL[i] < C[i] < RH[i]:
                side, stop, tgt = 1, L[i] * (1 - buffer_pct / 100), RH[i]
            elif H[i] > RH[i] > C[i] > RL[i]:
                side, stop, tgt = -1, H[i] * (1 + buffer_pct / 100), RL[i]
            if side:
                e = O[i + 1]
                if (side == 1 and stop < e < tgt) or (side == -1 and tgt < e < stop):
                    pos = {"side": side, "entry_time": w.index[i + 1], "entry": e, "stop": stop, "tgt": tgt}
                    # the entry bar itself is managed on the next loop pass (i + 1)
    if pos:
        trades.append({**pos, "exit_time": w.index[-1], "exit": C[-1], "reason": "end",
                       "R": r_mult(pos["side"], pos["entry"], pos["stop"], C[-1], cost_pct)})
    return trades


# ----------------------------------------------------------------- stats
def stats(tr, risk_pct=1.0):
    if tr.empty:
        return {"Trades": 0}
    r = tr["R"].to_numpy()
    eq_r = np.cumsum(r)
    dd_r = (eq_r - np.maximum.accumulate(np.maximum(eq_r, 0))).min()
    eq = np.cumprod(1 + r * risk_pct / 100)
    dd_pct = ((eq / np.maximum.accumulate(np.maximum(eq, 1))) - 1).min() * 100
    win, loss = r[r > 0].sum(), -r[r < 0].sum()
    hold = (tr["exit_time"] - tr["entry_time"]).dt.total_seconds().median() / 3600
    longs, shorts = tr[tr.side == 1], tr[tr.side == -1]
    return {
        "Trades": len(r),
        "Win %": round((r > 0).mean() * 100, 1),
        "Avg R": round(r.mean(), 3),
        "Total R": round(r.sum(), 1),
        "Profit factor": round(win / loss, 2) if loss else np.inf,
        "Max DD (R)": round(dd_r, 1),
        "Return @1% risk": f"{(eq[-1] - 1) * 100:.1f}%",
        "Max DD @1%": f"{dd_pct:.1f}%",
        "Long R": round(longs["R"].sum(), 1),
        "Short R": round(shorts["R"].sum(), 1),
        "Median hold (h)": round(hold, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-pct", type=float, default=0.06, help="round-trip cost, %% of price")
    ap.add_argument("--out", default="results/btc")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    h1 = load("1h.csv")
    h4 = resample(h1, "4h")
    d1 = load("daily.csv")
    d1_2y = resample(h1, "1D")                   # daily candles built from the same hourly data
    wk = d1.resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    span = lambda df: f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}"

    runs = [
        ("Zero Lag", "H1", span(h1), zero_lag(h1, a.cost_pct)),
        ("Zero Lag", "H4", span(h4), zero_lag(h4, a.cost_pct)),
        ("Zero Lag", "D1", span(d1), zero_lag(d1, a.cost_pct)),
        ("SMA 9/18 pullback", "H1", span(h1), sma_pullback(h1, a.cost_pct)),
        ("SMA 9/18 pullback", "H4", span(h4), sma_pullback(h4, a.cost_pct)),
        ("SMA 9/18 pullback", "D1", span(d1), sma_pullback(d1, a.cost_pct)),
        ("CRT", "H4 range → H1 entry", span(h1), crt(h1, h4, a.cost_pct)),
        ("CRT", "D1 range → H1 entry", span(h1), crt(h1, d1_2y, a.cost_pct)),
        ("CRT", "W1 range → D1 entry", span(d1), crt(d1, wk, a.cost_pct)),
    ]

    rows, all_trades, yearly = [], [], []
    for name, tf, period, trades in runs:
        tr = pd.DataFrame(trades)
        if not tr.empty:
            tr.insert(0, "strategy", name)
            tr.insert(1, "tf", tf)
            all_trades.append(tr)
            y = tr.groupby(tr["entry_time"].dt.year)["R"].sum().round(1)
            yearly.append(pd.Series(y, name=f"{name} {tf}"))
        rows.append({"Strategy": name, "Timeframe": tf, "Period": period, **stats(tr)})

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(all_trades).to_csv(out / "trades.csv", index=False)
    yr = pd.concat(yearly, axis=1).T.fillna(0)
    yr = yr[sorted(yr.columns)]
    yr.columns = [str(c) for c in yr.columns]
    yr.to_csv(out / "yearly_R.csv")

    md = [f"# BTC/USD backtest — Zero Lag vs SMA 9/18 pullback vs CRT\n",
          f"Data: Yahoo BTC-USD. H1/H4 = last ~2 years; D1/W1 = full history. "
          f"Cost {a.cost_pct}% of price per round trip. Results in R; same-bar SL+TP counted as SL.\n",
          summary.to_markdown(index=False), "\n\n## Total R by year\n", yr.to_markdown()]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
