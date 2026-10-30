import unittest
import pandas as pd
from scripts.stage_c_liquidity import liquidity_sweep_events, align_m15_liquidity_to_m5


class StageCLiquidityTests(unittest.TestCase):
    def test_sell_side_sweep_requires_wick_through_and_close_reclaim(self):
        high = [5, 4, 5, 4]
        low = [4, 2, 3, 1.8]
        close = [4, 3, 4, 2.2]
        out = liquidity_sweep_events(high, low, close, 1, 1)
        self.assertFalse(out.loc[2, "sell_side_sweep"])
        self.assertTrue(out.loc[3, "sell_side_sweep"])
        self.assertEqual(out.loc[3, "swept_swing_low"], 2)

    def test_buy_side_sweep_requires_wick_through_and_close_reclaim(self):
        high = [2, 5, 3, 5.2]
        low = [1, 2, 1, 2]
        close = [1.5, 4, 2, 4.5]
        out = liquidity_sweep_events(high, low, close, 1, 1)
        self.assertFalse(out.loc[2, "buy_side_sweep"])
        self.assertTrue(out.loc[3, "buy_side_sweep"])
        self.assertEqual(out.loc[3, "swept_swing_high"], 5)

    def test_sweep_must_reclaim_close_not_merely_touch_level(self):
        out = liquidity_sweep_events([5, 4, 5, 4], [4, 2, 3, 1.8],
                                     [4, 3, 4, 1.9], 1, 1)
        self.assertFalse(out.loc[3, "sell_side_sweep"])

    def test_m15_event_not_available_before_m15_close(self):
        times = pd.date_range("2025-01-01T00:00:00Z", periods=6, freq="5min")
        rows = []
        for i, ts in enumerate(times):
            p = 1.1 + i * 0.0001
            rows.append({"Datetime": ts, "BidHigh": p + .0002,
                         "BidLow": p - .0002, "BidClose": p,
                         "AskHigh": p + .0003, "AskLow": p - .0001,
                         "AskClose": p + .0001})
        out = align_m15_liquidity_to_m5(pd.DataFrame(rows), 1, 1, 3)
        self.assertTrue(out.loc[:2, "m15_close_time"].isna().all())
        self.assertEqual(out.loc[3, "m15_close_time"],
                         pd.Timestamp("2025-01-01T00:15:00Z"))

    def test_incomplete_m15_group_is_not_published(self):
        times = pd.date_range("2025-01-01T00:00:00Z", periods=6, freq="5min")
        rows = []
        for i, ts in enumerate(times):
            p = 1.1 + i * .0001
            rows.append({"Datetime": ts, "BidHigh": p+.0002,
                         "BidLow": p-.0002, "BidClose": p,
                         "AskHigh": p+.0003, "AskLow": p-.0001,
                         "AskClose": p+.0001})
        x = pd.DataFrame(rows).drop(index=1)
        out = align_m15_liquidity_to_m5(x, 1, 1, 3)
        self.assertFalse(out.loc[3, "m15_recent_sell_side_sweep"])
        self.assertTrue(pd.isna(out.loc[3, "m15_close_time"]))


if __name__ == "__main__":
    unittest.main()
