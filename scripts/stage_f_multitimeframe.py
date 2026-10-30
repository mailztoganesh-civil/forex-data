"""Stage F: causal H1 trend bias aligned to M5 execution bars.

Builds H1 candles from exactly twelve consecutive M5 BID/ASK midpoint bars.
EMA bias is published only after the H1 close; the M5 bar opening at that
timestamp is the first eligible bar. Missing/incomplete H1 groups are rejected.
"""
from __future__ import annotations
import pandas as pd

REQUIRED = ["Datetime", "BidOpen", "BidHigh", "BidLow", "BidClose",
            "AskOpen", "AskHigh", "AskLow", "AskClose"]


def h1_bias_events(open_, high, low, close, fast_period=20, slow_period=50):
    if fast_period < 1 or slow_period <= fast_period:
        raise ValueError("slow_period must be greater than positive fast_period")
    c = pd.Series(close).reset_index(drop=True).astype(float)
    fast = c.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean()
    slow = c.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean()
    direction = pd.Series(0, index=c.index, dtype="int64")
    direction[(c > fast) & (fast > slow)] = 1
    direction[(c < fast) & (fast < slow)] = -1
    return pd.DataFrame({"h1_ema_fast": fast, "h1_ema_slow": slow,
                         "h1_bias_direction": direction})


def align_h1_bias_to_m5(df, fast_period=20, slow_period=50):
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    x = df[REQUIRED].copy()
    x["Datetime"] = pd.to_datetime(x.Datetime, utc=True, errors="coerce")
    for col in REQUIRED[1:]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna().sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    if x.empty:
        return pd.DataFrame(columns=["Datetime", "h1_close_time", "h1_bias_direction"])
    for field in ["open", "high", "low", "close"]:
        x["mid_" + field] = (x["Bid" + field.title()] + x["Ask" + field.title()]) / 2
    x["slot"] = x.Datetime.dt.floor("1h")
    h = x.groupby("slot", sort=True).agg(
        count=("Datetime", "size"), times=("Datetime", lambda s: tuple(s)),
        open=("mid_open", "first"), high=("mid_high", "max"),
        low=("mid_low", "min"), close=("mid_close", "last")).reset_index()
    expected = h.slot.map(lambda t: tuple(t + pd.Timedelta(minutes=5*i) for i in range(12)))
    h = h[(h["count"] == 12) & (h.times == expected)].copy()
    h["h1_close_time"] = h.slot + pd.Timedelta(hours=1)
    if not h.empty:
        h = h.sort_values("slot").reset_index(drop=True)
        h["segment"] = h.slot.diff().ne(pd.Timedelta(hours=1)).cumsum()
        pieces = []
        for _, seg in h.groupby("segment", sort=False):
            ev = h1_bias_events(seg.open, seg.high, seg.low, seg.close,
                                fast_period, slow_period)
            part = seg[["h1_close_time", "segment"]].copy()
            part["h1_bias_direction"] = ev.h1_bias_direction.to_numpy()
            pieces.append(part)
        right = pd.concat(pieces, ignore_index=True).sort_values("h1_close_time")
        out = pd.merge_asof(x[["Datetime"]].sort_values("Datetime"), right.drop(columns="segment"),
                            left_on="Datetime", right_on="h1_close_time",
                            direction="backward", allow_exact_matches=True)
    else:
        out = x[["Datetime"]].copy()
        out["h1_close_time"] = pd.NaT
        out["h1_bias_direction"] = 0
    out["h1_bias_direction"] = out.h1_bias_direction.fillna(0).astype(int)
    return out
