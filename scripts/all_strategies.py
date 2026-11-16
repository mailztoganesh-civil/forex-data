"""
One fair test of all GR's strategies on currency, gold and BTC.

Markets / data (M5):
  EURUSD, EURJPY, XAUUSD : Dukascopy bid M5, Nov 2016 -> Nov 2026 (data/duka_m5)
  BTCUSD                 : Binance BTCUSDT M5, 2018 -> Nov 2026 (data/BTCUSD/5m)

Strategies (rules as in GR's bots / apps, default settings, nothing re-fitted per market):
  ZL_M5        Zero Lag bot defaults on M5: flip entry, band stop, 2R, close on flip,
               loss filter, FX: min SL 5 / max SL 50 pips
  ZL_M30_TRAIL Zero Lag BTC_M30_Trail preset: M30, H1 filter, band stop, 5xATR trail, no TP,
               no weekend entries
  ASIAN_BO     AsianBreakoutBot: range 03-07 UTC, first M5 close outside 07-14 UTC, stop range
               middle, 3R, flat 20:00 UTC, one trade a day (FX: min stop 3 pips)
  GOLD_BO      GoldBreakoutBot V2: range 01-07, entries 07-16, close beyond range +/-0.1 ATR,
               body >= 60%, close in outer 25%, H1 close vs H1 EMA50, stop middle, 2R, flat 20:00
  GOLD_SBS     GoldSbsBot: swing breakout sequence, swing 3, buy/sell-stop at P5, stop at sweep,
               3R, max sequence 150 bars, time exit 300 bars
  EMA_34_50    EmaCrossBot: EMA 34/50 cross 07-16 UTC, stop 3x ATR14, 3R, max 3 a day, flat 20:00
  SMA_PULLBACK sma-signals engine: SMA 9/18 cross, 10-candle impulse, SMA 9 pullback, 60% body
               bounce, stop order, swing stop, 2R, 1h SMA50 trend, sideways filter, 07-17 UTC
               (app default is M3; tested on M5)
  CRT_H1_M5    eurusd-crt-backtest: previous H1 range swept by an M5 candle that closes back
               inside, London 08-11 / New York 08-11 local time, stop beyond the wick, target the
               opposite side of the H1 range

Costs per round trip: EURUSD 0.9 pip, EURJPY 1.2 pip, XAUUSD $0.25, BTC 0.06% of price.
R = (side*(exit-entry) - cost) / (|entry-stop| + cost). Sharpe: daily P/L at 1% risk per trade,
calendar days, x sqrt(365). Three periods per market to check consistency.

Usage: python scripts/all_strategies.py [--out results/all_strategies] [--only EURUSD,...]
"""

import argparse
import glob
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import zl_research2 as zr  # noqa: E402

UTC = "UTC"
MARKETS = {
    "EURUSD": dict(kind="fx", pip=0.0001, cost=0.00009),
    "EURJPY": dict(kind="fx", pip=0.01, cost=0.012),
    "XAUUSD": dict(kind="gold", pip=0.1, cost=0.25),
    "BTCUSD": dict(kind="crypto", pip=None, cost_pct=0.06),
}
PERIODS = {
    "fx": [("2016-11-15", "2020-01-01"), ("2020-01-01", "2023-01-01"), ("2023-01-01", None)],
    "gold": [("2016-11-15", "2020-01-01"), ("2020-01-01", "2023-01-01"), ("2023-01-01", None)],
    "crypto": [("2018-01-01", "2020-01-01"), ("2020-01-01", "2023-01-01"), ("2023-01-01", None)],
}


# ------------------------------------------------------------------ data
def load(sym):
    if sym == "BTCUSD":
        df = zr.load_5m_vol()[["open", "high", "low", "close"]]
    else:
        parts = [pd.read_csv(f, header=None, names=["t", "open", "high", "low", "close"])
                 for f in sorted(glob.glob(f"data/duka_m5/{sym}_m5_*.csv.gz"))]
        df = pd.concat(parts).drop_duplicates("t").sort_values("t")
        df.index = pd.to_datetime(df.pop("t"), unit="s", utc=True)
    return df.astype(float)


def rs(df, rule):
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def wilder_atr(H, L, C, n=14):
    prev = np.r_[C[0], C[:-1]]
    tr = np.maximum(H - L, np.maximum(np.abs(H - prev), np.abs(L - prev)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def mk(trades, idx):
    t = pd.DataFrame(trades, columns=["ji", "jo", "side", "entry", "stop", "exit", "reason"])
    t["entry_time"] = idx[t["ji"].to_numpy()] if len(t) else []
    t["exit_time"] = idx[t["jo"].to_numpy()] if len(t) else []
    return t.drop(columns=["ji", "jo"])


def bar_exit(side, o, h, l, stop, tgt):
    """SL first if both touched (conservative); gaps fill at the open."""
    if side == 1:
        if l <= stop:
            return min(o, stop), "SL"
        if tgt is not None and h >= tgt:
            return max(o, tgt), "TP"
    else:
        if h >= stop:
            return max(o, stop), "SL"
        if tgt is not None and l <= tgt:
            return min(o, tgt), "TP"
    return None, None


# ------------------------------------------------------------------ strategies
def s_zl_m5(df, m):
    ctx = zr.Ctx2(df.assign(volume=0.0), df.assign(volume=0.0))
    p = {**zr.BASE, "mtf": None}
    if m["kind"] == "fx":
        p.update(min_sl=5 * m["pip"], max_sl=50 * m["pip"])
    t = zr.run2(ctx, p, 0.0)
    return t[["side", "entry", "stop", "exit", "reason", "entry_time", "exit_time"]]


def s_zl_trail(df, m):
    m30 = rs(df, "30min")
    ctx = zr.Ctx2(m30.assign(volume=0.0), df.assign(volume=0.0))
    p = {**zr.BASE, "trail_atr": 5.0, "rr": 0.0, "no_weekend": True}
    t = zr.run2(ctx, p, 0.0)
    return t[["side", "entry", "stop", "exit", "reason", "entry_time", "exit_time"]]


def session_breakout(df, m, r0, r1, last_entry, rr, v2=False, max_range=0.0, min_range=0.0):
    O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    idx = df.index
    n = len(C)
    day = idx.normalize()
    hour = idx.hour.to_numpy()
    mins = (idx.hour * 60 + idx.minute).to_numpy()
    atr = wilder_atr(H, L, C)
    if v2:
        h1 = rs(df, "1h")
        ema = h1["close"].ewm(span=50, adjust=False).mean()
        # last COMPLETED H1 candle at the close of each M5 bar
        h1c = pd.Series(h1["close"].to_numpy(), index=h1.index + pd.Timedelta("1h"))
        h1e = pd.Series(ema.to_numpy(), index=h1.index + pd.Timedelta("1h"))
        close_t = idx + pd.Timedelta("5min")
        hc = h1c.reindex(close_t, method="ffill").to_numpy()
        he = h1e.reindex(close_t, method="ffill").to_numpy()
    out, pos = [], None
    cur, rhi, rlo, ready, done, skip = None, 0, 0, False, False, False
    cnt, hi, lo = 0, -np.inf, np.inf
    for j in range(1, n):
        i = j - 1                                   # last closed bar, acting at open of j
        if day[j] != cur:
            cur, ready, done, skip, cnt, hi, lo = day[j], False, False, False, 0, -np.inf, np.inf
        # manage open position (flat at the first bar opening >= 20:00)
        if pos is not None and hour[j] >= 20:
            out.append((pos["j"], j, pos["side"], pos["e"], pos["s"], O[j], "time"))
            pos = None
        if pos is not None:
            px, why = bar_exit(pos["side"], O[j] if j != pos["j"] else pos["e"], H[j], L[j], pos["s"], pos["t"])
            if px is not None:
                out.append((pos["j"], j, pos["side"], pos["e"], pos["s"], px, why))
                pos = None
        # range accumulation uses bar i (same day)
        if day[i] == cur and r0 * 60 <= mins[i] < r1 * 60:
            cnt += 1
            hi, lo = max(hi, H[i]), min(lo, L[i])
        if not ready and not skip and day[i] == cur and mins[i] + 5 >= r1 * 60:
            if cnt < (r1 - r0) * 12 * 0.8:
                skip = True
            elif (max_range > 0 and hi - lo > max_range) or (min_range > 0 and hi - lo < min_range):
                skip = True
            else:
                rhi, rlo, ready = hi, lo, True
        if pos is not None or done or skip or not ready or hour[j] >= 20:
            continue
        if not (mins[i] >= r1 * 60 and mins[i] + 5 <= last_entry * 60 and day[i] == cur):
            continue
        buf = 0.1 * atr[i - 1] if v2 else 0.0
        side = 1 if C[i] > rhi + buf else -1 if C[i] < rlo - buf else 0
        if not side:
            continue
        if v2:
            span = H[i] - L[i]
            if span <= 0 or abs(C[i] - O[i]) < 0.6 * span:
                continue
            if (side == 1 and C[i] < H[i] - 0.25 * span) or (side == -1 and C[i] > L[i] + 0.25 * span):
                continue
            if np.isnan(hc[i]) or (side == 1 and hc[i] <= he[i]) or (side == -1 and hc[i] >= he[i]):
                continue
        done = True
        e = O[j]
        s = (rhi + rlo) / 2
        if side * (e - s) <= 0:
            continue
        if m["kind"] == "fx" and abs(e - s) < 3 * m["pip"]:
            continue
        pos = {"j": j, "side": side, "e": e, "s": s, "t": e + side * rr * abs(e - s)}
        px, why = bar_exit(side, e, H[j], L[j], s, pos["t"])
        if px is not None:
            out.append((j, j, side, e, s, px, why))
            pos = None
    return mk(out, df.index)


def s_asian(df, m):
    return session_breakout(df, m, 3, 7, 14, 3.0)


def s_gold_bo(df, m):
    return session_breakout(df, m, 1, 7, 16, 2.0, v2=True)


def s_ema(df, m, fast=34, slow=50, atr_mult=3.0, rr=3.0):
    O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    idx = df.index
    f = pd.Series(C).ewm(span=fast, adjust=False).mean().to_numpy()
    s_ = pd.Series(C).ewm(span=slow, adjust=False).mean().to_numpy()
    atr = wilder_atr(H, L, C)
    day = idx.normalize()
    hour = idx.hour.to_numpy()
    mins = (idx.hour * 60 + idx.minute).to_numpy()
    out, pos, cur, nday = [], None, None, 0
    for j in range(slow + 2, len(C)):
        i = j - 1
        if day[j] != cur:
            cur, nday = day[j], 0
        if pos is not None and hour[j] >= 20:
            out.append((pos["j"], j, pos["side"], pos["e"], pos["s"], O[j], "time"))
            pos = None
        if pos is not None:
            px, why = bar_exit(pos["side"], O[j] if j != pos["j"] else pos["e"], H[j], L[j], pos["s"], pos["t"])
            if px is not None:
                out.append((pos["j"], j, pos["side"], pos["e"], pos["s"], px, why))
                pos = None
        if pos is not None or hour[j] >= 20 or not (7 * 60 <= mins[i] < 16 * 60):
            continue
        side = 1 if (f[i] > s_[i] and f[i - 1] <= s_[i - 1]) else -1 if (f[i] < s_[i] and f[i - 1] >= s_[i - 1]) else 0
        if not side or nday >= 3:
            continue
        nday += 1
        e = O[j]
        st = e - side * atr_mult * atr[i]
        pos = {"j": j, "side": side, "e": e, "s": st, "t": e + side * rr * abs(e - st)}
        px, why = bar_exit(side, e, H[j], L[j], st, pos["t"])
        if px is not None:
            out.append((j, j, side, e, st, px, why))
            pos = None
    return mk(out, idx)


def s_sma(df, m, N=10, rr=2.0, c_max=1, c_look=50):
    O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    idx = df.index
    n = len(C)
    buf = m["pip"] if m["pip"] else None
    s9 = pd.Series(C).rolling(9).mean().to_numpy()
    s18 = pd.Series(C).rolling(18).mean().to_numpy()
    Rng = pd.Series(H - L).rolling(N).sum().to_numpy()
    # hourly trend: sign(previous hour close - SMA50 of hourly closes at previous hour)
    h1 = df["close"].resample("1h", label="left", closed="left").last().dropna()
    sm = h1.rolling(50).mean()
    tr_h = np.sign(h1.shift(1) - sm.shift(1)).fillna(0)
    trend = tr_h.reindex(idx.floor("1h")).fillna(0).to_numpy().astype(int)
    hour = idx.hour.to_numpy()
    cross = np.zeros(n, dtype=int)
    for i in range(1, n):
        x = (s9[i - 1] <= s18[i - 1] and s9[i] > s18[i]) or (s9[i - 1] >= s18[i - 1] and s9[i] < s18[i])
        cross[i] = cross[i - 1] + int(x)
    out, pos = [], None
    s_dir = st = 0
    ref = need = ext = sw = 0.0
    p_entry = p_stop = None
    for i in range(1, n):
        if pos is not None:
            px, why = bar_exit(pos["side"], O[i] if pos["j"] != i else pos["e"], H[i], L[i], pos["s"], pos["t"])
            if px is not None:
                out.append((pos["j"], i, pos["side"], pos["e"], pos["s"], px, why))
                pos = None
            else:
                continue
        up = i >= 19 and s9[i - 1] <= s18[i - 1] and s9[i] > s18[i]
        dn = i >= 19 and s9[i - 1] >= s18[i - 1] and s9[i] < s18[i]
        if (up and s_dir == 1) or (dn and s_dir == -1):
            up = dn = False
        if up or dn:
            d = 1 if up else -1
            p_entry = p_stop = None
            chop_ok = cross[i] - cross[max(0, i - c_look)] - 1 <= c_max
            if trend[i] == d and chop_ok:
                s_dir, st, ref, need, ext = d, 1, C[i], Rng[i], C[i]
            else:
                s_dir = st = 0
            continue
        if not s_dir:
            continue
        if p_entry is not None:
            trig = H[i] > p_entry if s_dir == 1 else L[i] < p_entry
            if trig and 7 <= hour[i] < 17:
                fill = max(O[i], p_entry) if s_dir == 1 else min(O[i], p_entry)
                pos = {"j": i, "side": s_dir, "e": fill, "s": p_stop, "t": fill + s_dir * rr * abs(fill - p_stop)}
                s_dir = st = 0
                p_entry = p_stop = None
                px, why = bar_exit(pos["side"], fill, H[i], L[i], pos["s"], pos["t"])
                if px is not None:
                    out.append((i, i, pos["side"], fill, pos["s"], px, why))
                    pos = None
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
                b = buf if buf else C[i] * 0.0002
                p_entry = H[i] if s_dir == 1 else L[i]
                p_stop = sw - s_dir * b
    return mk(out, idx)


def s_crt(df, m):
    h1 = rs(df, "1h")
    rng = h1[["high", "low"]].shift(1).rename(columns={"high": "rh", "low": "rl"})
    w = df.join(rng.reindex(df.index, method="ffill")).dropna()
    O, H, L, C = (w[c].to_numpy() for c in ("open", "high", "low", "close"))
    RH, RL = w["rh"].to_numpy(), w["rl"].to_numpy()
    lon = w.index.tz_convert(ZoneInfo("Europe/London")).hour.to_numpy()
    ny = w.index.tz_convert(ZoneInfo("America/New_York")).hour.to_numpy()
    sess = ((lon >= 8) & (lon < 11)) | ((ny >= 8) & (ny < 11))
    buf_abs = 0.2 * m["pip"] if m["pip"] else None
    out, pos = [], None
    for i in range(len(w) - 1):
        if pos is not None:
            px, why = bar_exit(pos["side"], O[i] if pos["j"] != i else pos["e"], H[i], L[i], pos["s"], pos["t"])
            if px is not None:
                out.append((pos["j"], i, pos["side"], pos["e"], pos["s"], px, why))
                pos = None
        if pos is None and sess[i]:
            b = buf_abs if buf_abs else C[i] * 0.0002
            side = 0
            if L[i] < RL[i] < C[i] < RH[i]:
                side, st, tg = 1, L[i] - b, RH[i]
            elif H[i] > RH[i] > C[i] > RL[i]:
                side, st, tg = -1, H[i] + b, RL[i]
            if side:
                e = O[i + 1]
                if (side == 1 and st < e < tg) or (side == -1 and tg < e < st):
                    pos = {"j": i + 1, "side": side, "e": e, "s": st, "t": tg}
    return mk(out, w.index)


def s_sbs(df, m, N=3, rr=3.0, max_bars=150):
    O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
    n = len(C)

    def hd(i, d): return H[i] if d == 1 else -L[i]
    def ld(i, d): return L[i] if d == 1 else -H[i]
    def cd(i, d): return d * C[i]

    def piv_h(j, d):
        if j - N < 0:
            return False
        v = hd(j, d)
        return all(hd(k, d) < v for k in range(j - N, j)) and all(hd(k, d) <= v for k in range(j + 1, j + N + 1))

    def piv_l(j, d):
        if j - N < 0:
            return False
        v = ld(j, d)
        return all(ld(k, d) > v for k in range(j - N, j)) and all(ld(k, d) >= v for k in range(j + 1, j + N + 1))

    def new():
        return {"S": 0, "P1": 0, "Origin": 0.0, "P2": 0.0, "Hpre": 0.0, "P3": 0.0, "Sw": 0, "L4": 0.0, "L4i": 0,
                "Trig": np.nan}

    seq = {1: new(), -1: new()}
    sh_idx, sh_lvl = {1: -1, -1: -1}, {1: 0.0, -1: 0.0}
    out, pos = [], None
    pend = {}                                     # d -> (entry, stop) active from next bar
    for i in range(2 * N + 2, n - 1):
        # --- bar i: pending fills and position management happen DURING bar i
        if pos is None and pend:
            for d, (en, sl) in list(pend.items()):
                hit = (H[i] >= en) if d == 1 else (L[i] <= en)
                if hit:
                    fill = max(O[i], en) if d == 1 else min(O[i], en)
                    pos = {"j": i, "side": d, "e": fill, "s": sl, "t": fill + d * rr * abs(fill - sl)}
                    seq[d] = new()
                    pend = {}
                    break
        if pos is not None:
            if i - pos["j"] >= 2 * max_bars:
                out.append((pos["j"], i, pos["side"], pos["e"], pos["s"], O[i], "time"))
                pos = None
            else:
                px, why = bar_exit(pos["side"], O[i] if pos["j"] != i else pos["e"], H[i], L[i], pos["s"], pos["t"])
                if px is not None:
                    out.append((pos["j"], i, pos["side"], pos["e"], pos["s"], px, why))
                    pos = None
        # --- bar i closed: update sequences (same as the bot's Step)
        for d in (1, -1):
            s = seq[d]
            j = i - N
            if j >= N and piv_h(j, d) and s["S"] == 0:
                sh_idx[d], sh_lvl[d] = j, hd(j, d)
            if s["S"] > 0 and (ld(i, d) < s["Origin"] or i - s["P1"] > max_bars):
                seq[d] = s = new()
                pend.pop(d, None)
            if s["S"] == 0:
                if sh_idx[d] >= 0 and cd(i, d) > sh_lvl[d]:
                    j0 = sh_idx[d]
                    s.update(S=1, P1=i, Origin=min(ld(k, d) for k in range(j0, i + 1)))
                    sh_idx[d] = -1
            elif s["S"] == 1:
                if j > s["P1"] and piv_l(j, d):
                    s.update(S=2, P2=ld(j, d), Hpre=max(hd(k, d) for k in range(s["P1"], i + 1)))
            elif s["S"] == 2:
                if ld(i, d) < s["P2"]:
                    seq[d] = new()
                    pend.pop(d, None)
                elif hd(i, d) >= s["Hpre"]:
                    s.update(S=3, P3=hd(i, d))
            elif s["S"] == 3:
                if hd(i, d) > s["P3"]:
                    s["P3"] = hd(i, d)
                if ld(i, d) < s["P2"]:
                    s.update(S=4, Sw=i, L4=ld(i, d), L4i=i, Trig=np.nan)
            elif s["S"] == 4:
                if ld(i, d) < s["L4"]:
                    s.update(L4=ld(i, d), L4i=i, Trig=np.nan)
                    pend.pop(d, None)
                if j > s["L4i"] and piv_h(j, d):
                    s["Trig"] = hd(j, d)
                if i - s["Sw"] > max_bars:
                    seq[d] = new()
                    pend.pop(d, None)
        # --- manage pending orders (only while flat)
        if pos is None:
            for d in (1, -1):
                s = seq[d]
                if s["S"] == 4 and not np.isnan(s["Trig"]):
                    en, sl = d * s["Trig"], d * s["L4"]
                    if abs(en - sl) > 0:
                        pend[d] = (en, sl)
                else:
                    pend.pop(d, None)
        else:
            pend = {}
    return mk(out, df.index)


STRATS = {
    "ZL_M5": s_zl_m5, "ZL_M30_TRAIL": s_zl_trail, "ASIAN_BO": s_asian, "GOLD_BO": s_gold_bo,
    "GOLD_SBS": s_sbs, "EMA_34_50": s_ema, "SMA_PULLBACK": s_sma, "CRT_H1_M5": s_crt,
}


# ------------------------------------------------------------------ metrics
def add_r(t, m):
    if t.empty:
        t["R"] = []
        return t
    cost = m["cost"] if "cost" in m else t["entry"] * m["cost_pct"] / 100
    t["R"] = (t["side"] * (t["exit"] - t["entry"]) - cost) / ((t["entry"] - t["stop"]).abs() + cost)
    return t


def stats(t, a, b):
    a = pd.Timestamp(a, tz=UTC)
    b = pd.Timestamp(b, tz=UTC) if b else None
    tt = t[(t["entry_time"] >= a) & ((t["entry_time"] < b) if b is not None else True)]
    end = b or (t["exit_time"].max() if len(t) else a)
    days = pd.date_range(a.normalize(), end.normalize(), freq="D", tz=UTC, inclusive="left")
    if tt.empty or len(days) < 30:
        return dict(trades=0, sharpe=np.nan, totR=0.0, pf=np.nan, win=np.nan, cagr=np.nan, maxdd=np.nan, ddR=0.0)
    d = tt.set_index("exit_time")["R"].resample("D").sum().reindex(days, fill_value=0.0) / 100
    r = tt["R"].to_numpy()
    eq = (1 + d).cumprod()
    cum = np.cumsum(r)
    loss = -r[r < 0].sum()
    return dict(trades=len(r), sharpe=round(d.mean() / d.std() * np.sqrt(365), 2) if d.std() > 0 else np.nan,
                totR=round(r.sum(), 1), pf=round(r[r > 0].sum() / loss, 2) if loss else np.inf,
                win=round((r > 0).mean() * 100, 1), cagr=round((eq.iloc[-1] ** (365 / len(d)) - 1) * 100, 1),
                maxdd=round(((eq / eq.cummax()) - 1).min() * 100, 1),
                ddR=round((cum - np.maximum.accumulate(np.maximum(cum, 0))).min(), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/all_strategies")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, trades = [], []
    for sym, m in MARKETS.items():
        if a.only and sym not in a.only.split(","):
            continue
        df = load(sym)
        pers = PERIODS[m["kind"]]
        df = df[df.index >= pd.Timestamp(pers[0][0], tz=UTC) - pd.Timedelta(days=60)]
        print(f"{sym}: {len(df)} bars {df.index[0]} → {df.index[-1]}", flush=True)
        for name, fn in STRATS.items():
            t = add_r(fn(df, m), m)
            t = t[t["entry_time"] >= pd.Timestamp(pers[0][0], tz=UTC)]
            full = stats(t, pers[0][0], None)
            row = {"market": sym, "strategy": name, **{k: full[k] for k in
                   ("trades", "sharpe", "totR", "pf", "win", "cagr", "maxdd", "ddR")}}
            for k, (pa, pb) in enumerate(pers, start=1):
                row[f"P{k} sharpe"] = stats(t, pa, pb)["sharpe"]
            row["all periods > 0"] = all((row[f"P{k} sharpe"] or -1) > 0 for k in (1, 2, 3))
            rows.append(row)
            trades.append(t.assign(market=sym, strategy=name))
            print(f"  {name:13s} n={full['trades']:5d} Sharpe {full['sharpe']} totR {full['totR']} "
                  f"P1 {row['P1 sharpe']} P2 {row['P2 sharpe']} P3 {row['P3 sharpe']}", flush=True)
    res = pd.DataFrame(rows).sort_values(["market", "sharpe"], ascending=[True, False])
    res.to_csv(out / "ranking.csv", index=False)
    pd.concat(trades).to_csv(out / "all_trades.csv.gz", index=False, compression="gzip")
    md = ["# All strategies on currency, gold and BTC\n",
          "Same costs model and rules for every strategy (bot defaults). Sharpe = daily P/L at 1% risk per trade, "
          "x sqrt(365). FX & gold: Dukascopy M5 Nov 2016 → Nov 2026 (P1 2016-19, P2 2020-22, P3 2023-26). "
          "BTC: Binance M5 2018 → Nov 2026 (P1 2018-19, P2 2020-22, P3 2023-26). "
          "CAGR / max DD at 1% risk.\n"]
    for sym in res["market"].unique():
        md += [f"\n## {sym}\n", res[res.market == sym].drop(columns=["market"]).to_markdown(index=False)]
    (out / "ranking.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
