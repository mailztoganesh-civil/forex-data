import unittest
import pandas as pd
from scripts.stage_f_multitimeframe import h1_bias_events, align_h1_bias_to_m5


class StageFTests(unittest.TestCase):
    def test_bias_requires_ordered_ema_and_price(self):
        closes = list(range(1, 80))
        out = h1_bias_events(closes, closes, closes, closes, 3, 8)
        self.assertEqual(out.iloc[-1].h1_bias_direction, 1)
        down = list(range(80, 1, -1))
        out = h1_bias_events(down, down, down, down, 3, 8)
        self.assertEqual(out.iloc[-1].h1_bias_direction, -1)

    def test_h1_bias_not_visible_before_hour_close(self):
        times = pd.date_range("2025-01-01T00:00:00Z", periods=24, freq="5min")
        rows = []
        for i, ts in enumerate(times):
            p = 1.1 + i * 0.0001
            rows.append({"Datetime": ts, "BidOpen": p, "BidHigh": p+.0002,
                         "BidLow": p-.0002, "BidClose": p,
                         "AskOpen": p+.0001, "AskHigh": p+.0003,
                         "AskLow": p-.0001, "AskClose": p+.0001})
        out = align_h1_bias_to_m5(pd.DataFrame(rows), 2, 3)
        self.assertTrue(out.loc[:11, "h1_close_time"].isna().all())
        self.assertEqual(out.loc[12, "h1_close_time"],
                         pd.Timestamp("2025-01-01T01:00:00Z"))

    def test_incomplete_hour_is_not_published(self):
        times = pd.date_range("2025-01-01T00:00:00Z", periods=24, freq="5min")
        rows = []
        for i, ts in enumerate(times):
            p = 1.1 + i * .0001
            rows.append({"Datetime": ts, "BidOpen": p, "BidHigh": p+.0002,
                         "BidLow": p-.0002, "BidClose": p,
                         "AskOpen": p+.0001, "AskHigh": p+.0003,
                         "AskLow": p-.0001, "AskClose": p+.0001})
        out = align_h1_bias_to_m5(pd.DataFrame(rows).drop(index=4), 2, 3)
        self.assertEqual(out.loc[12, "h1_bias_direction"], 0)
        self.assertTrue(pd.isna(out.loc[12, "h1_close_time"]))


if __name__ == "__main__":
    unittest.main()
