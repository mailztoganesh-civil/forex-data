"""Stage D: causal M15 fair-value-gap (FVG) retest confirmation."""
from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED = ["Datetime", "BidHigh", "BidLow", "BidClose",
            "AskHigh", "AskLow", "AskClose"]


def fvg_retest_events(high, low, close, open_, max_age=20, min_gap=0.0):
    """Detect three-candle gaps, then a later retest with directional rejection.

    Bullish FVG: current low > high two candles earlier. A later candle must
    overlap the zone and close above its midpoint and above its open.
    Bearish FVG: current high < low two candles earlier; inverse rejection.
    Each gap can trigger once. All decisions use completed candles only.
    """
    if max_age < 1 or min_gap < 0:
        raise ValueError("max_age must be >= 1 and min_gap >= 0")
    h = pd.Series(high).reset_index(drop=True).astype(float)
    l = pd.Series(low).reset_index(drop=True).astype(float)
    c = pd.Series(close).reset_index(drop=True).astype(float)
    o = pd.Series(open_).reset_index(drop=True).astype(float)
    n = len(c)
    out = pd.DataFrame({
        "bullish_fvg_retest": False, "bearish_fvg_retest": False,
        "fvg_lower": np.nan, "fvg_upper": np.nan,
        "fvg_age": np.nan,
    }, index=range(n))
    zones = []
    for i in range(n):
        active = []
        for z in zones:
            age = i - z["created"]
            if age > max_age:
                continue
            overlap = l.iloc[i] <= z["upper"] and h.iloc[i] >= z["lower"]
            if overlap:
                midpoint = (z["lower"] + z["upper"]) / 2
                if z["side"] == "bull" and c.iloc[i] > midpoint and c.iloc[i] > o.iloc[i]:
                    out.at[i, "bullish_fvg_retest"] = True
                    out.at[i, "fvg_lower"] = z["lower"]
                    out.at[i, "fvg_upper"] = z["upper"]
                    out.at[i, "fvg_age"] = age
                    z["used"] = True
                elif z["side"] == "bear" and c.iloc[i] < midpoint and c.iloc[i] < o.iloc[i]:
                    out.at[i, "bearish_fvg_retest"] = True
                    out.at[i, "fvg_lower"] = z["lower"]
                    out.at[i, "fvg_upper"] = z["upper"]
                    out.at[i, "fvg_age"] = age
                    z["used"] = True
            if not z.get("used", False):
                active.append(z)
        zones = active
        # A gap is only knowable once its third candle has closed.
        if i >= 2:
            bull_size = l.iloc[i] - h.iloc[i-2]
            bear_size = l.iloc[i-2] - h.iloc[i]
            if bull_size > min_gap:
                zones.append({"side":"bull", "lower":float(h.iloc[i-2]),
                              "upper":float(l.iloc[i]), "created":i})
            if bear_size > min_gap:
                zones.append({"side":"bear", "lower":float(h.iloc[i]),
                              "upper":float(l.iloc[i-2]), "created":i})
    return out


def align_m15_fvg_to_m5(df, max_age=20, min_gap=0.0, recent_bars=3):
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
    cols = ["m15_bullish_fvg_retest", "m15_bearish_fvg_retest",
            "m15_recent_bullish_fvg_retest", "m15_recent_bearish_fvg_retest"]
    if x.empty:
        return pd.DataFrame(columns=["Datetime", "m15_close_time", *cols])
    x["mid_open"] = (x.BidClose.shift(0) * 0 + (x.BidClose + x.AskClose)/2)
    x["mid_high"] = (x.BidHigh + x.AskHigh)/2
    x["mid_low"] = (x.BidLow + x.AskLow)/2
    x["mid_close"] = (x.BidClose + x.AskClose)/2
    x["slot"] = x.Datetime.dt.floor("15min")
    m15 = x.groupby("slot", sort=True).agg(
        count=("Datetime","size"), times=("Datetime",lambda s: tuple(s)),
        open=("mid_open","first"), high=("mid_high","max"),
        low=("mid_low","min"), close=("mid_close","last")).reset_index()
    expected = m15.slot.map(lambda t:(t,t+pd.Timedelta(minutes=5),t+pd.Timedelta(minutes=10)))
    m15 = m15[(m15["count"]==3)&(m15.times==expected)].copy()
    m15["m15_close_time"] = m15.slot + pd.Timedelta(minutes=15)
    chunks=[]
    if not m15.empty:
        m15=m15.sort_values("slot").reset_index(drop=True)
        m15["segment"]=m15.slot.diff().ne(pd.Timedelta(minutes=15)).cumsum()
        for _,seg in m15.groupby("segment",sort=False):
            ev=fvg_retest_events(seg.high,seg.low,seg.close,seg.open,max_age,min_gap)
            part=seg[["slot","m15_close_time","segment"]].copy()
            part["m15_bullish_fvg_retest"]=ev.bullish_fvg_retest.to_numpy()
            part["m15_bearish_fvg_retest"]=ev.bearish_fvg_retest.to_numpy()
            chunks.append(part)
    if chunks:
        right=pd.concat(chunks,ignore_index=True).sort_values("m15_close_time")
        right["m15_recent_bullish_fvg_retest"]=right.groupby("segment").m15_bullish_fvg_retest.transform(
            lambda s:s.rolling(recent_bars,min_periods=1).max()).astype(bool)
        right["m15_recent_bearish_fvg_retest"]=right.groupby("segment").m15_bearish_fvg_retest.transform(
            lambda s:s.rolling(recent_bars,min_periods=1).max()).astype(bool)
        right=right.drop(columns=["slot","segment"])
        out=pd.merge_asof(x[["Datetime"]],right,left_on="Datetime",right_on="m15_close_time",
                          direction="backward",allow_exact_matches=True)
    else:
        out=x[["Datetime"]].copy()
        out["m15_close_time"]=pd.NaT
        for col in cols: out[col]=False
    for col in cols: out[col]=out[col].fillna(False).astype(bool)
    return out
