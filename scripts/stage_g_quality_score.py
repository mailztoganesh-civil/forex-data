"""Stage G directional confirmation scoring for Phantom Flow entries.

The Stage A trigger remains mandatory. Confirmations are additive evidence,
not mandatory gates: H1 alignment is worth 2 points; M15 structure, liquidity
sweep/reclaim, FVG retest and order-block retest are each worth 1 point.
All inputs must already be causally aligned to the M5 entry bar.
"""
def directional_confirmation_score(side, h1_direction=0, m15_direction=0,
                                    sell_side_sweep=False, buy_side_sweep=False,
                                    bullish_fvg=False, bearish_fvg=False,
                                    bullish_ob=False, bearish_ob=False):
    if side not in ("long", "short"):
        return {"score": 0, "h1": 0, "structure": 0,
                "liquidity": 0, "fvg": 0, "order_block": 0}
    sign = 1 if side == "long" else -1
    h1 = 2 if int(h1_direction) == sign else 0
    structure = 1 if int(m15_direction) == sign else 0
    liquidity = 1 if (sell_side_sweep if side == "long" else buy_side_sweep) else 0
    fvg = 1 if (bullish_fvg if side == "long" else bearish_fvg) else 0
    ob = 1 if (bullish_ob if side == "long" else bearish_ob) else 0
    return {"score": h1 + structure + liquidity + fvg + ob,
            "h1": h1, "structure": structure, "liquidity": liquidity,
            "fvg": fvg, "order_block": ob}
