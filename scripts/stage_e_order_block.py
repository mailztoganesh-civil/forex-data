"""Stage E: causal M15 order-block formation and retest confirmation."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = ["Datetime", "BidOpen", "BidHigh", "BidLow", "BidClose",
            "AskOpen", "AskHigh", "AskLow", "AskClose"]


def order_block_retest_events(open_, high, low, close, lookback=8,
                              atr_period=14, displacement_atr=1.0,
                              max_age=20):
    """Detect last opposing candle before a confirmed displacement/BOS.

    Bullish OB: bullish impulse closes above the prior lookback high, has a
    body >= displacement_atr * ATR, and is preceded by a bearish candle.
    Bearish OB is inverse. The nearest opposing candle in the lookback is
    used as the zone. Retest must occur on a later completed candle, overlap
    the zone, and close directionally beyond its midpoint. Each zone triggers
    once. All rolling breakout references are shifted to avoid look-ahead.
    """
    if lookback < 1 or atr_period < 1 or max_age < 1 or displacement_atr < 0:
        raise ValueError("lookback, atr_period, max_age must be positive; displacement_atr non-negative")
    o=pd.Series(open_).reset_index(drop=True).astype(float)
    h=pd.Series(high).reset_index(drop=True).astype(float)
    l=pd.Series(low).reset_index(drop=True).astype(float)
    c=pd.Series(close).reset_index(drop=True).astype(float)
    prev=c.shift(1)
    tr=pd.concat([(h-l).abs(),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/atr_period,adjust=False,min_periods=atr_period).mean()
    prior_high=h.shift(1).rolling(lookback,min_periods=lookback).max()
    prior_low=l.shift(1).rolling(lookback,min_periods=lookback).min()
    out=pd.DataFrame({"bullish_ob_retest":np.zeros(len(c), dtype=bool),
                      "bearish_ob_retest":np.zeros(len(c), dtype=bool),
                      "ob_lower":np.full(len(c), np.nan),
                      "ob_upper":np.full(len(c), np.nan),
                      "ob_age":np.full(len(c), np.nan)})
    zones=[]
    for i in range(len(c)):
        active=[]
        for z in zones:
            age=i-z["created"]
            if age>max_age:
                continue
            overlap=l.iat[i]<=z["upper"] and h.iat[i]>=z["lower"]
            if overlap:
                mid=(z["lower"]+z["upper"])/2
                if z["side"]=="bull" and c.iat[i]>mid and c.iat[i]>o.iat[i]:
                    out.at[i,"bullish_ob_retest"]=True
                    out.at[i,"ob_lower"]=z["lower"]; out.at[i,"ob_upper"]=z["upper"]
                    out.at[i,"ob_age"]=age; z["used"]=True
                elif z["side"]=="bear" and c.iat[i]<mid and c.iat[i]<o.iat[i]:
                    out.at[i,"bearish_ob_retest"]=True
                    out.at[i,"ob_lower"]=z["lower"]; out.at[i,"ob_upper"]=z["upper"]
                    out.at[i,"ob_age"]=age; z["used"]=True
            if not z.get("used",False):
                active.append(z)
        zones=active
        # Formation only after impulse candle closes; retests begin next bar.
        if i < max(lookback,atr_period)+1 or not np.isfinite(atr.iat[i]):
            continue
        body=abs(c.iat[i]-o.iat[i])
        if c.iat[i]>o.iat[i] and c.iat[i]>prior_high.iat[i] and body>=displacement_atr*atr.iat[i]:
            candidates=[j for j in range(max(0,i-lookback),i)
                        if c.iat[j]<o.iat[j]]
            if candidates:
                j=candidates[-1]
                zones.append({"side":"bull","lower":float(l.iat[j]),
                              "upper":float(h.iat[j]),"created":i})
        elif c.iat[i]<o.iat[i] and c.iat[i]<prior_low.iat[i] and body>=displacement_atr*atr.iat[i]:
            candidates=[j for j in range(max(0,i-lookback),i)
                        if c.iat[j]>o.iat[j]]
            if candidates:
                j=candidates[-1]
                zones.append({"side":"bear","lower":float(l.iat[j]),
                              "upper":float(h.iat[j]),"created":i})
    return out


def align_m15_order_blocks_to_m5(df, lookback=8, atr_period=14,
                                  displacement_atr=1.0, max_age=20,
                                  recent_bars=3):
    if recent_bars<1:
        raise ValueError("recent_bars must be >= 1")
    missing=set(REQUIRED)-set(df.columns)
    if missing:
        raise ValueError("Missing columns: "+", ".join(sorted(missing)))
    x=df[REQUIRED].copy()
    x["Datetime"]=pd.to_datetime(x.Datetime,utc=True,errors="coerce")
    for col in REQUIRED[1:]:
        x[col]=pd.to_numeric(x[col],errors="coerce")
    x=x.dropna().sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    cols=["m15_bullish_ob_retest","m15_bearish_ob_retest",
          "m15_recent_bullish_ob_retest","m15_recent_bearish_ob_retest"]
    if x.empty:
        return pd.DataFrame(columns=["Datetime","m15_close_time",*cols])
    x["mid_open"]=(x.BidOpen+x.AskOpen)/2
    x["mid_high"]=(x.BidHigh+x.AskHigh)/2
    x["mid_low"]=(x.BidLow+x.AskLow)/2
    x["mid_close"]=(x.BidClose+x.AskClose)/2
    x["slot"]=x.Datetime.dt.floor("15min")
    m=x.groupby("slot",sort=True).agg(
        count=("Datetime","size"),times=("Datetime",lambda s:tuple(s)),
        open=("mid_open","first"),high=("mid_high","max"),
        low=("mid_low","min"),close=("mid_close","last")).reset_index()
    expected=m.slot.map(lambda t:(t,t+pd.Timedelta(minutes=5),t+pd.Timedelta(minutes=10)))
    m=m[(m["count"]==3)&(m.times==expected)].copy()
    m["m15_close_time"]=m.slot+pd.Timedelta(minutes=15)
    chunks=[]
    if not m.empty:
        m=m.sort_values("slot").reset_index(drop=True)
        m["segment"]=m.slot.diff().ne(pd.Timedelta(minutes=15)).cumsum()
        for _,seg in m.groupby("segment",sort=False):
            ev=order_block_retest_events(seg.open,seg.high,seg.low,seg.close,
                lookback,atr_period,displacement_atr,max_age)
            part=seg[["slot","m15_close_time","segment"]].copy()
            part["m15_bullish_ob_retest"]=ev.bullish_ob_retest.to_numpy()
            part["m15_bearish_ob_retest"]=ev.bearish_ob_retest.to_numpy()
            chunks.append(part)
    if chunks:
        right=pd.concat(chunks,ignore_index=True).sort_values("m15_close_time")
        for side in ["bullish","bearish"]:
            col=f"m15_{side}_ob_retest"
            right[f"m15_recent_{side}_ob_retest"]=right.groupby("segment")[col].transform(
                lambda s:s.rolling(recent_bars,min_periods=1).max()).astype(bool)
        right=right.drop(columns=["slot","segment"])
        out=pd.merge_asof(x[["Datetime"]],right,left_on="Datetime",
            right_on="m15_close_time",direction="backward",allow_exact_matches=True)
    else:
        out=x[["Datetime"]].copy(); out["m15_close_time"]=pd.NaT
        for col in cols: out[col]=False
    for col in cols: out[col]=out[col].fillna(False).astype(bool)
    return out
