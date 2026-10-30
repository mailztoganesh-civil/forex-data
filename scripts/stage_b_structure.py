"""Non-repainting swing structure events (Stage B).

A pivot at bar i is only published at i + right_bars, after the right-side
confirmation bars have closed. Breaks are close-based and occur no earlier
than that confirmation time. This deliberately trades some signal delay for
causal, reproducible structure labels.
"""
import numpy as np
import pandas as pd


def structure_events(high, low, close, left_bars=3, right_bars=3):
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be >= 1")
    h = pd.Series(high).reset_index(drop=True).astype(float)
    l = pd.Series(low).reset_index(drop=True).astype(float)
    c = pd.Series(close).reset_index(drop=True).astype(float)
    n = len(c)
    out = pd.DataFrame({
        "swing_high_confirmed": np.nan,
        "swing_low_confirmed": np.nan,
        "bos_bull": False, "bos_bear": False,
        "choch_bull": False, "choch_bear": False,
        "structure_direction": 0
    }, index=range(n))
    last_high = last_low = np.nan
    high_used = low_used = False
    trend = 0
    for i in range(n):
        p = i - right_bars
        if p >= left_bars:
            a, b = p-left_bars, p+right_bars+1
            win_h = h.iloc[a:b]
            win_l = l.iloc[a:b]
            # Strict unique extreme avoids ambiguous equal-high/low pivots.
            if h.iloc[p] == win_h.max() and (win_h == h.iloc[p]).sum() == 1:
                last_high, high_used = float(h.iloc[p]), False
                out.at[i, "swing_high_confirmed"] = last_high
            if l.iloc[p] == win_l.min() and (win_l == l.iloc[p]).sum() == 1:
                last_low, low_used = float(l.iloc[p]), False
                out.at[i, "swing_low_confirmed"] = last_low
        # Breaks are evaluated after any newly confirmed pivots on this bar.
        if np.isfinite(last_high) and not high_used and c.iloc[i] > last_high:
            if trend < 0:
                out.at[i, "choch_bull"] = True
            else:
                out.at[i, "bos_bull"] = True
            trend, high_used = 1, True
        if np.isfinite(last_low) and not low_used and c.iloc[i] < last_low:
            if trend > 0:
                out.at[i, "choch_bear"] = True
            else:
                out.at[i, "bos_bear"] = True
            trend, low_used = -1, True
        out.at[i, "structure_direction"] = trend
    return out
