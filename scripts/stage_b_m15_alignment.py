"""Causal M15 structure state mapped onto M5 execution bars.

M5 timestamps are candle-open UTC timestamps. A complete M15 candle is built
from exactly three consecutive M5 candles; its structure events become
available only at the M15 close timestamp. merge_asof therefore exposes that
state to the M5 candle opening at/after the close, never to an earlier candle.
"""
from __future__ import annotations
import pandas as pd
from scripts.stage_b_structure import structure_events

REQUIRED = ["Datetime", "BidHigh", "BidLow", "BidClose",
            "AskHigh", "AskLow", "AskClose"]


def align_m15_structure_to_m5(df, left_bars=3, right_bars=3):
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    x = df[REQUIRED].copy()
    x["Datetime"] = pd.to_datetime(x["Datetime"], utc=True, errors="coerce")
    for col in REQUIRED[1:]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = (x.dropna().sort_values("Datetime")
           .drop_duplicates("Datetime").reset_index(drop=True))
    if x.empty:
        return pd.DataFrame(columns=["Datetime", "m15_close_time",
            "m15_structure_direction", "m15_bos_bull", "m15_bos_bear",
            "m15_choch_bull", "m15_choch_bear"])

    # Mid-price candles; timestamp is the M5 candle OPEN.
    x["mid_high"] = (x.BidHigh + x.AskHigh) / 2
    x["mid_low"] = (x.BidLow + x.AskLow) / 2
    x["mid_close"] = (x.BidClose + x.AskClose) / 2
    x["slot"] = x.Datetime.dt.floor("15min")
    grouped = x.groupby("slot", sort=True)
    m15 = grouped.agg(
        count=("Datetime", "size"),
        first_time=("Datetime", "min"),
        last_time=("Datetime", "max"),
        high=("mid_high", "max"), low=("mid_low", "min"),
        close=("mid_close", "last")).reset_index()
    # Reject incomplete bars and gaps: expected opens are :00, :05, :10.
    expected_last = m15.slot + pd.Timedelta(minutes=10)
    m15 = m15[(m15["count"] == 3) &
              (m15.first_time == m15.slot) &
              (m15.last_time == expected_last)].copy()
    m15["m15_close_time"] = m15.slot + pd.Timedelta(minutes=15)
    if m15.empty:
        result = x[["Datetime"]].copy()
        result["m15_close_time"] = pd.NaT
        for col, default in [("m15_structure_direction", 0),
            ("m15_bos_bull", False), ("m15_bos_bear", False),
            ("m15_choch_bull", False), ("m15_choch_bear", False)]:
            result[col] = default
        return result

    ev = structure_events(m15.high, m15.low, m15.close,
                          left_bars, right_bars)
    m15 = m15.reset_index(drop=True)
    for col in ["structure_direction", "bos_bull", "bos_bear",
                "choch_bull", "choch_bear"]:
        m15["m15_" + col] = ev[col].to_numpy()
    right = m15[["m15_close_time", "m15_structure_direction",
                 "m15_bos_bull", "m15_bos_bear",
                 "m15_choch_bull", "m15_choch_bear"]].sort_values("m15_close_time")
    # M5 candle opening at the M15 close is the first eligible execution bar.
    left = x[["Datetime"]].sort_values("Datetime")
    out = pd.merge_asof(left, right, left_on="Datetime",
                        right_on="m15_close_time", direction="backward",
                        allow_exact_matches=True)
    out["m15_structure_direction"] = out.m15_structure_direction.fillna(0).astype(int)
    for col in ["m15_bos_bull", "m15_bos_bear",
                "m15_choch_bull", "m15_choch_bear"]:
        out[col] = out[col].fillna(False).astype(bool)
    return out
