#!/usr/bin/env python3
"""Stage A Phantom-style ATR shift + MA oscillator backtest.

Input: audited EURUSD BID/ASK M5 CSV with Datetime, BidOpen/High/Low/Close,
AskOpen/High/Low/Close. Signals are computed on midpoint OHLC and confirmed
at bar close. Orders fill at the next bar open on the executable quote side.
If stop and target are both touched in one bar, stop is assumed first.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PRICE = ["BidOpen","BidHigh","BidLow","BidClose","AskOpen","AskHigh","AskLow","AskClose"]

def wilder_atr(h, l, c, n):
    prev = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def supertrend_direction(h, l, c, atr, factor):
    mid = (h+l)/2
    upper = mid + factor*atr
    lower = mid - factor*atr
    final_u = upper.copy()
    final_l = lower.copy()
    direction = pd.Series(index=c.index, dtype="float64")
    for i in range(len(c)):
        if i == 0 or not np.isfinite(atr.iat[i]):
            direction.iat[i] = np.nan
            continue
        if i == 1 or not np.isfinite(direction.iat[i-1]):
            direction.iat[i] = 1.0
        else:
            if not (upper.iat[i] < final_u.iat[i-1] or c.iat[i-1] > final_u.iat[i-1]):
                final_u.iat[i] = final_u.iat[i-1]
            if not (lower.iat[i] > final_l.iat[i-1] or c.iat[i-1] < final_l.iat[i-1]):
                final_l.iat[i] = final_l.iat[i-1]
            prev_dir = direction.iat[i-1]
            if prev_dir > 0:
                direction.iat[i] = -1.0 if c.iat[i] > final_u.iat[i] else 1.0
            else:
                direction.iat[i] = 1.0 if c.iat[i] < final_l.iat[i] else -1.0
    return direction

def indicators(df, a):
    mid_o=(df.BidOpen+df.AskOpen)/2
    h=(df.BidHigh+df.AskHigh)/2
    l=(df.BidLow+df.AskLow)/2
    c=(df.BidClose+df.AskClose)/2
    atr=wilder_atr(h,l,c,a.atr)
    d=supertrend_direction(h,l,c,atr,a.factor)
    fast=c.ewm(span=a.osc,adjust=False,min_periods=a.osc).mean()
    base=c.rolling(a.base,min_periods=a.base).mean()
    osc=((fast-base)/atr.replace(0,np.nan)).ewm(span=a.smooth,adjust=False,min_periods=a.smooth).mean()
    bull_shift=(d<0)&(d.shift(1)>0)
    bear_shift=(d>0)&(d.shift(1)<0)
    bull_cross=(osc>a.threshold)&(osc.shift(1)<=a.threshold)
    bear_cross=(osc< -a.threshold)&(osc.shift(1)>=-a.threshold)
    df=df.copy()
    df["ATR"]=atr; df["Osc"]=osc
    df["LongSignal"]=(bull_shift&bull_cross).fillna(False)
    df["ShortSignal"]=(bear_shift&bear_cross).fillna(False)
    df["MidClose"]=c
    return df

def run_backtest(df,a):
    start=pd.Timestamp(a.start,tz="UTC")
    end=pd.Timestamp(a.end,tz="UTC")+pd.Timedelta(days=1)
    df=df[(df.Datetime>=start)&(df.Datetime<end)].reset_index(drop=True)
    trades=[]; pos=None
    slip=a.slippage_pips*0.0001
    comm=a.commission_pips*0.0001
    for i in range(1,len(df)):
        r=df.iloc[i]
        # New-bar entry from the prior, fully closed signal bar.
        prev=df.iloc[i-1]
        if pos is not None:
            if pos["side"]=="long":
                stop_hit=r.BidLow <= pos["stop"]
                target_hit=r.BidHigh >= pos["target"]
                if stop_hit or target_hit:
                    # Conservative tie-break: stop first.
                    px=(pos["stop"]-slip) if stop_hit else (pos["target"]-slip)
                    pnl=(px-pos["entry"])-2*comm
                    trades.append({**pos,"exit_time":r.Datetime,"exit_price":px,
                                   "exit_reason":"stop" if stop_hit else "target","pnl":pnl})
                    pos=None
            else:
                stop_hit=r.AskHigh >= pos["stop"]
                target_hit=r.AskLow <= pos["target"]
                if stop_hit or target_hit:
                    px=(pos["stop"]+slip) if stop_hit else (pos["target"]+slip)
                    pnl=(pos["entry"]-px)-2*comm
                    trades.append({**pos,"exit_time":r.Datetime,"exit_price":px,
                                   "exit_reason":"stop" if stop_hit else "target","pnl":pnl})
                    pos=None
        if pos is None and np.isfinite(prev.ATR) and prev.ATR>0:
            side="long" if prev.LongSignal else ("short" if prev.ShortSignal else None)
            if side:
                entry=(r.AskOpen+slip) if side=="long" else (r.BidOpen-slip)
                risk=prev.ATR*a.stop_atr
                if side=="long":
                    pos={"side":side,"entry_time":r.Datetime,"entry":entry,
                         "stop":entry-risk,"target":entry+risk*a.rr,
                         "signal_time":prev.Datetime}
                else:
                    pos={"side":side,"entry_time":r.Datetime,"entry":entry,
                         "stop":entry+risk,"target":entry-risk*a.rr,
                         "signal_time":prev.Datetime}
                # Entry is at this bar's open; process same-bar excursion.
                if side=="long":
                    sh=r.BidLow<=pos["stop"]; th=r.BidHigh>=pos["target"]
                    if sh or th:
                        px=(pos["stop"]-slip) if sh else (pos["target"]-slip)
                        trades.append({**pos,"exit_time":r.Datetime,"exit_price":px,
                                       "exit_reason":"stop" if sh else "target",
                                       "pnl":(px-entry)-2*comm})
                        pos=None
                else:
                    sh=r.AskHigh>=pos["stop"]; th=r.AskLow<=pos["target"]
                    if sh or th:
                        px=(pos["stop"]+slip) if sh else (pos["target"]+slip)
                        trades.append({**pos,"exit_time":r.Datetime,"exit_price":px,
                                       "exit_reason":"stop" if sh else "target",
                                       "pnl":(entry-px)-2*comm})
                        pos=None
    if pos is not None:
        r=df.iloc[-1]
        px=(r.BidClose-slip) if pos["side"]=="long" else (r.AskClose+slip)
        pnl=((px-pos["entry"]) if pos["side"]=="long" else (pos["entry"]-px))-2*comm
        trades.append({**pos,"exit_time":r.Datetime,"exit_price":px,"exit_reason":"end_of_test","pnl":pnl})
    t=pd.DataFrame(trades)
    if t.empty:
        return t, {"trades":0,"win_rate_pct":None,"profit_factor":None,"expectancy_pips":None,"max_drawdown_pips":None,"net_pips":0.0}
    t["pnl_pips"]=t.pnl/0.0001
    t["equity_pips"]=t.pnl_pips.cumsum()
    peak=t.equity_pips.cummax().clip(lower=0)
    dd=t.equity_pips-peak
    wins=t.loc[t.pnl_pips>0,"pnl_pips"].sum()
    losses=-t.loc[t.pnl_pips<0,"pnl_pips"].sum()
    summary={"trades":int(len(t)),"win_rate_pct":round((t.pnl_pips>0).mean()*100,2),
             "profit_factor":round(wins/losses,3) if losses else None,
             "expectancy_pips":round(t.pnl_pips.mean(),3),
             "max_drawdown_pips":round(dd.min(),3),
             "net_pips":round(t.pnl_pips.sum(),3)}
    return t,summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",default="reports/stage_a")
    p.add_argument("--start",default="2021-01-01")
    p.add_argument("--end",default="2026-12-31")
    p.add_argument("--atr",type=int,default=10)
    p.add_argument("--factor",type=float,default=3.0)
    p.add_argument("--osc",type=int,default=15)
    p.add_argument("--base",type=int,default=40)
    p.add_argument("--smooth",type=int,default=3)
    p.add_argument("--threshold",type=float,default=0.5)
    p.add_argument("--stop-atr",type=float,default=1.5)
    p.add_argument("--rr",type=float,default=2.0)
    p.add_argument("--slippage-pips",type=float,default=0.2)
    p.add_argument("--commission-pips",type=float,default=0.0)
    a=p.parse_args()
    df=pd.read_csv(a.input)
    missing=set(["Datetime",*PRICE])-set(df.columns)
    if missing: raise SystemExit("Missing columns: "+", ".join(sorted(missing)))
    df["Datetime"]=pd.to_datetime(df.Datetime,utc=True,errors="coerce")
    for c in PRICE: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["Datetime",*PRICE]).sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    df=indicators(df,a)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    results=[]
    for label,start,end in [("IS_2021_2024","2021-01-01","2024-12-31"),("OOS_2025_2026","2025-01-01",a.end)]:
        aa=argparse.Namespace(**vars(a)); aa.start=start; aa.end=end
        trades,summary=run_backtest(df,aa)
        trades.to_csv(out/f"{label.lower()}_trades.csv",index=False)
        results.append({"period":label,**summary})
    pd.DataFrame(results).to_csv(out/"summary.csv",index=False)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Reports written to {out}")

if __name__=="__main__": main()
