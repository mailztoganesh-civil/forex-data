"""Stage C: causal M15 liquidity sweep and reclaim events.

A sell-side sweep pierces the latest already-confirmed swing low intrabar and
closes back above it. A buy-side sweep pierces the latest already-confirmed
swing high and closes back below it. A swing only becomes usable after its
right-side pivot confirmation; events are published at the M15 close.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ["Datetime", "BidHigh", "BidLow", "BidClose",
            "AskHigh", "AskLow", "AskClose"]


def liquidity_sweep_events(high, low, close, left_bars=3, right_bars=3):
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be >= 1")
    h = pd.Series(high).reset_index(drop=True).astype(float)
    l = pd.Series(low).reset_index(drop=True).astype(float)
    c = pd.Series(close).reset_index(drop=True).astype(float)
    n = len(c)
    out = pd.DataFrame({
        "sell_side_sweep": False, "buy_side_sweep": False,
        "swept_swing_low": np.nan, "swept_swing_high": np.nan,
    }, index=range(n))
    last_high = last_low = np.nan
    for i in range(n):
        # Test against levels that were known before this candle began.
        if np.isfinite(last_low) and l.iloc[i] < last_low and c.iloc[i] > last_low:
            out.at[i, "sell_side_sweep"] = True
            out.at[i, "swept_swing_low"] = last_low
        if np.isfinite(last_high) and h.iloc[i] > last_high and c.iloc[i] < last_high:
            out.at[i, "buy_side_sweep"] = True
            out.at[i, "swept_swing_high"] = last_high

        # Newly confirmed pivots are only available after this candle closes,
        # so they become eligible starting with the next candle.
        p = i - right_bars
        if p >= left_bars:
            a, b = p - left_bars, p + right_bars + 1
            wh, wl = h.iloc[a:b], l.iloc[a:b]
            if h.iloc[p] == wh.max() and (wh == h.iloc[p]).sum() == 1:
                last_high = float(h.iloc[p])
            if l.iloc[p] == wl.min() and (wl == l.iloc[p]).sum() == 1:
                last_low = float(l.iloc[p])
    return out


def align_m15_liquidity_to_m5(df, left_bars=3, right_bars=3, recent_bars=3):
    """Return one row per input M5 row; expose M15 sweeps only after close."""
    if recent_bars < 1:
        raise ValueError("recent_bars must be >= 1")
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    x = df[REQUIRED].copy()
    x["Datetime"] = pd.to_datetime(x["Datetime"], utc=True, errors="coerce")
    for col in REQUIRED[1:]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna().sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    cols = ["m15_sell_side_sweep", "m15_buy_side_sweep",
            "m15_recent_sell_side_sweep", "m15_recent_buy_side_sweep"]
    if x.empty:
        return pd.DataFrame(columns=["Datetime", "m15_close_time", *cols])

    x["mid_high"] = (x.BidHigh + x.AskHigh) / 2
    x["mid_low"] = (x.BidLow + x.AskLow) / 2
    x["mid_close"] = (x.BidClose + x.AskClose) / 2
    x["slot"] = x.Datetime.dt.floor("15min")
    m15 = x.groupby("slot", sort=True).agg(
        count=("Datetime", "size"),
        times=("Datetime", lambda s: tuple(s)),
        high=("mid_high", "max"), low=("mid_low", "min"),
        close=("mid_close", "last")).reset_index()
    expected = m15.slot.map(lambda t: (t, t + pd.Timedelta(minutes=5),
                                        t + pd.Timedelta(minutes=10)))
    m15 = m15[(m15["count"] == 3) & (m15.times == expected)].copy()
    m15["m15_close_time"] = m15.slot + pd.Timedelta(minutes=15)
    events = []
    if not m15.empty:
        m15 = m15.sort_values("slot").reset_index(drop=True)
        m15["segment"] = m15.slot.diff().ne(pd.Timedelta(minutes=15)).cumsum()
        for _, seg in m15.groupby("segment", sort=False):
            ev = liquidity_sweep_events(seg.high, seg.low, seg.close,
                                        left_bars, right_bars)
            part = seg[["slot", "m15_close_time", "segment"]].copy()
            part["m15_sell_side_sweep"] = ev.sell_side_sweep.to_numpy()
            part["m15_buy_side_sweep"] = ev.buy_side_sweep.to_numpy()
            events.append(part)
    if events:
        right = pd.concat(events, ignore_index=True).sort_values("m15_close_time")
        right["m15_recent_sell_side_sweep"] = right.groupby("segment").m15_sell_side_sweep.transform(
            lambda s: s.rolling(recent_bars, min_periods=1).max()).astype(bool)
        right["m15_recent_buy_side_sweep"] = right.groupby("segment").m15_buy_side_sweep.transform(
            lambda s: s.rolling(recent_bars, min_periods=1).max()).astype(bool)
        right = right.drop(columns="segment")
        left = x[["Datetime"]].sort_values("Datetime")
        out = pd.merge_asof(left, right.drop(columns="slot"),
                            left_on="Datetime", right_on="m15_close_time",
                            direction="backward", allow_exact_matches=True)
    else:
        out = x[["Datetime"]].copy()
        out["m15_close_time"] = pd.NaT
        for col in cols:
            out[col] = False
    for col in cols:
        out[col] = out[col].fillna(False).astype(bool)
    return out
