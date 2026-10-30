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
from scripts.stage_b_m15_alignment import align_m15_structure_to_m5
from scripts.stage_c_liquidity import align_m15_liquidity_to_m5
from scripts.stage_d_fvg import align_m15_fvg_to_m5
from scripts.stage_e_order_block import align_m15_order_blocks_to_m5
from scripts.stage_f_multitimeframe import align_h1_bias_to_m5
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

def order_block_allows(side, recent_bullish_retest, recent_bearish_retest):
    """Longs need bullish OB rejection; shorts need bearish OB rejection."""
    if side == "long":
        return bool(recent_bullish_retest)
    if side == "short":
        return bool(recent_bearish_retest)
    return False


def fvg_allows(side, recent_bullish_retest, recent_bearish_retest):
    """Longs need bullish FVG rejection; shorts need bearish FVG rejection."""
    if side == "long":
        return bool(recent_bullish_retest)
    if side == "short":
        return bool(recent_bearish_retest)
    return False


def liquidity_allows(side, recent_sell_sweep, recent_buy_sweep):
    """Longs need a recent sell-side sweep; shorts need a recent buy-side sweep."""
    if side == "long":
        return bool(recent_sell_sweep)
    if side == "short":
        return bool(recent_buy_sweep)
    return False


def h1_bias_allows(side, direction):
    """Require the confirmed H1 trend bias to agree with the M5 trigger."""
    if side == "long":
        return int(direction) > 0
    if side == "short":
        return int(direction) < 0
    return False


def structure_allows(side, direction):
    """Allow entries only when confirmed M15 direction agrees with the trade."""
    if side == "long":
        return int(direction) > 0
    if side == "short":
        return int(direction) < 0
    return False


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
        had_position_at_bar_open = pos is not None
        if pos is not None:
            if pos["side"]=="long":
                stop_hit=r.BidLow <= pos["stop"]
                target_hit=r.BidHigh >= pos["target"]
                if stop_hit or target_hit:
                    # Conservative tie-break: stop first.
                    px=(min(pos["stop"], r.BidOpen)-slip) if stop_hit else (pos["target"]-slip)
                    pnl=(px-pos["entry"])-2*comm
                    trades.append({**pos,"exit_time":r.Datetime,"exit_price":px,
                                   "exit_reason":"stop" if stop_hit else "target","pnl":pnl})
                    pos=None
            else:
                stop_hit=r.AskHigh >= pos["stop"]
                target_hit=r.AskLow <= pos["target"]
                if stop_hit or target_hit:
                    px=(max(pos["stop"], r.AskOpen)+slip) if stop_hit else (pos["target"]+slip)
                    pnl=(pos["entry"]-px)-2*comm
                    trades.append({**pos,"exit_time":r.Datetime,"exit_price":px,
                                   "exit_reason":"stop" if stop_hit else "target","pnl":pnl})
                    pos=None
        # Never enter at this bar's open after an existing position exited
        # later inside the same bar; that would be a retroactive fill.
        if (not had_position_at_bar_open) and pos is None and np.isfinite(prev.ATR) and prev.ATR>0:
            side="long" if prev.LongSignal else ("short" if prev.ShortSignal else None)
            if side and getattr(a, "liquidity_filter", False):
                if not liquidity_allows(side, r.m15_recent_sell_side_sweep,
                                        r.m15_recent_buy_side_sweep):
                    side = None
            if side and getattr(a, "order_block_filter", False):
                if not order_block_allows(side, r.m15_recent_bullish_ob_retest,
                                          r.m15_recent_bearish_ob_retest):
                    side = None
            if side and getattr(a, "fvg_filter", False):
                if not fvg_allows(side, r.m15_recent_bullish_fvg_retest,
                                  r.m15_recent_bearish_fvg_retest):
                    side = None
            if side and getattr(a, "h1_filter", False):
                if not h1_bias_allows(side, r.h1_bias_direction):
                    side = None
            if side and getattr(a, "structure_filter", False):
                # Structure is checked at the actual entry bar open. This
                # includes an M15 candle that closed at this exact timestamp.
                if not structure_allows(side, r.m15_structure_direction):
                    side = None
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
    p.add_argument("--liquidity-recent-bars",type=int,default=3,
                   help="M15 bars after a sweep during which Stage C may confirm an entry")
    p.add_argument("--fvg-max-age",type=int,default=20,
                   help="Maximum M15 bars between FVG formation and retest")
    p.add_argument("--fvg-min-gap-pips",type=float,default=0.0,
                   help="Minimum FVG width in EURUSD pips")
    p.add_argument("--fvg-recent-bars",type=int,default=3,
                   help="M15 bars during which a confirmed FVG retest may confirm entry")
    p.add_argument("--ob-lookback",type=int,default=8,
                   help="M15 bars searched for the last opposing order-block candle")
    p.add_argument("--ob-atr-period",type=int,default=14)
    p.add_argument("--ob-displacement-atr",type=float,default=1.0,
                   help="Minimum impulse body in ATR multiples")
    p.add_argument("--ob-max-age",type=int,default=20)
    p.add_argument("--ob-recent-bars",type=int,default=3)
    p.add_argument("--h1-ema-fast",type=int,default=20)
    p.add_argument("--h1-ema-slow",type=int,default=50)
    a=p.parse_args()
    df=pd.read_csv(a.input)
    missing=set(["Datetime",*PRICE])-set(df.columns)
    if missing: raise SystemExit("Missing columns: "+", ".join(sorted(missing)))
    df["Datetime"]=pd.to_datetime(df.Datetime,utc=True,errors="coerce")
    for c in PRICE: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["Datetime",*PRICE]).sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    source_start = df.Datetime.min()
    source_end = df.Datetime.max()
    df=indicators(df,a)
    h1 = align_h1_bias_to_m5(df, fast_period=a.h1_ema_fast,
                             slow_period=a.h1_ema_slow)
    df = df.merge(h1.drop(columns=["Datetime"]), left_index=True,
                  right_index=True, how="left")
    df["h1_bias_direction"] = df["h1_bias_direction"].fillna(0).astype(int)
    # Compute confirmed M15 structure once on the full chronological dataset,
    # then make it available only to M5 bars opening at/after M15 close.
    aligned=align_m15_structure_to_m5(df)
    df=df.merge(aligned.drop(columns=["Datetime"]), left_index=True,
                right_index=True, how="left")
    df["m15_structure_direction"]=df["m15_structure_direction"].fillna(0).astype(int)
    liquidity = align_m15_liquidity_to_m5(df, recent_bars=a.liquidity_recent_bars)
    df = df.merge(liquidity.drop(columns=["Datetime"]), left_index=True,
                  right_index=True, how="left")
    fvg = align_m15_fvg_to_m5(df, max_age=a.fvg_max_age,
                              min_gap=a.fvg_min_gap_pips*0.0001,
                              recent_bars=a.fvg_recent_bars)
    df = df.merge(fvg.drop(columns=["Datetime"]), left_index=True,
                  right_index=True, how="left")
    ob = align_m15_order_blocks_to_m5(df, lookback=a.ob_lookback,
        atr_period=a.ob_atr_period, displacement_atr=a.ob_displacement_atr,
        max_age=a.ob_max_age, recent_bars=a.ob_recent_bars)
    df = df.merge(ob.drop(columns=["Datetime"]), left_index=True,
                  right_index=True, how="left")
    for col in ["m15_bullish_ob_retest", "m15_bearish_ob_retest",
                "m15_recent_bullish_ob_retest", "m15_recent_bearish_ob_retest"]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in ["m15_bullish_fvg_retest", "m15_bearish_fvg_retest",
                "m15_recent_bullish_fvg_retest", "m15_recent_bearish_fvg_retest"]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in ["m15_sell_side_sweep", "m15_buy_side_sweep",
                "m15_recent_sell_side_sweep", "m15_recent_buy_side_sweep"]:
        df[col] = df[col].fillna(False).astype(bool)
    for col in ["m15_bos_bull","m15_bos_bear","m15_choch_bull","m15_choch_bear"]:
        df[col]=df[col].fillna(False).astype(bool)

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    results=[]
    variants = [
        ("A_BASELINE", False, False, False, False),
        ("A_PLUS_B_M15", True, False, False, False),
        ("A_PLUS_C_LIQUIDITY", False, True, False, False),
        ("A_PLUS_B_PLUS_C", True, True, False, False),
        ("A_PLUS_D_FVG", False, False, True, False),
        ("A_PLUS_B_PLUS_C_PLUS_D", True, True, True, False),
        ("A_PLUS_E_ORDER_BLOCK", False, False, False, True),
        ("A_PLUS_B_PLUS_C_PLUS_D_PLUS_E", True, True, True, True, False),
        ("A_PLUS_F_H1_TREND", False, False, False, False, True),
        ("A_PLUS_B_PLUS_F", True, False, False, False, True),
    ]
    for variant, use_structure, use_liquidity, use_fvg, use_ob, use_h1 in variants:
        for label,start,end in [("IS_2021_2024","2021-01-01","2024-12-31"),("OOS_2025_2026","2025-01-01",a.end)]:
            aa=argparse.Namespace(**vars(a)); aa.start=start; aa.end=end
            aa.structure_filter=use_structure
            aa.liquidity_filter=use_liquidity
            aa.fvg_filter=use_fvg
            aa.order_block_filter=use_ob
            aa.h1_filter=use_h1
            trades,summary=run_backtest(df,aa)
            trades.to_csv(out/f"{variant.lower()}_{label.lower()}_trades.csv",index=False)
            period_start = pd.Timestamp(start, tz="UTC")
            period_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            available = df[(df.Datetime >= period_start) & (df.Datetime <= period_end)]
            coverage_start = available.Datetime.min() if not available.empty else pd.NaT
            coverage_end = available.Datetime.max() if not available.empty else pd.NaT
            coverage_complete = bool(
                not available.empty and coverage_start <= period_start + pd.Timedelta(days=1)
                and coverage_end >= period_end - pd.Timedelta(days=1)
            )
            results.append({"variant":variant,"period":label,
                            "input_coverage_start_utc":source_start,
                            "input_coverage_end_utc":source_end,
                            "period_coverage_start_utc":coverage_start,
                            "period_coverage_end_utc":coverage_end,
                            "period_coverage_complete":coverage_complete,**summary})
    summary_df=pd.DataFrame(results)
    summary_df.to_csv(out/"comparison_summary.csv",index=False)
    print(summary_df.to_string(index=False))
    print(f"Reports written to {out}")

if __name__=="__main__": main()
