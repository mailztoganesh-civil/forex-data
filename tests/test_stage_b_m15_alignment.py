import unittest
import pandas as pd
from scripts.stage_b_m15_alignment import align_m15_structure_to_m5


class M15AlignmentTests(unittest.TestCase):
    def make_data(self, n=18):
        times = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="5min")
        rows = []
        for i, ts in enumerate(times):
            # A deterministic gentle rise; all quote sides are valid.
            p = 1.10 + i * 0.0001
            rows.append({"Datetime": ts, "BidHigh": p + .0002,
                "BidLow": p - .0002, "BidClose": p,
                "AskHigh": p + .0003, "AskLow": p - .0001,
                "AskClose": p + .0001})
        return pd.DataFrame(rows)

    def test_state_is_not_visible_before_first_m15_close(self):
        out = align_m15_structure_to_m5(self.make_data())
        self.assertTrue(pd.isna(out.loc[0, "m15_close_time"]))
        self.assertTrue(pd.isna(out.loc[1, "m15_close_time"]))
        self.assertTrue(pd.isna(out.loc[2, "m15_close_time"]))
        self.assertEqual(out.loc[3, "m15_close_time"],
                         pd.Timestamp("2025-01-01T00:15:00Z"))

    def test_incomplete_m15_group_is_not_published(self):
        x = self.make_data()
        x = x[x.Datetime != pd.Timestamp("2025-01-01T00:10:00Z")]
        out = align_m15_structure_to_m5(x)
        self.assertTrue(pd.isna(out.loc[0, "m15_close_time"]))
        # Next complete quarter-hour is available only at its close.
        row = out[out.Datetime == pd.Timestamp("2025-01-01T00:30:00Z")].iloc[0]
        self.assertEqual(row.m15_close_time,
                         pd.Timestamp("2025-01-01T00:30:00Z"))

    def test_irregular_three_row_group_is_rejected(self):
        x = self.make_data()
        x.loc[x.Datetime == pd.Timestamp("2025-01-01T00:05:00Z"),
              "Datetime"] = pd.Timestamp("2025-01-01T00:06:00Z")
        out = align_m15_structure_to_m5(x)
        row = out[out.Datetime == pd.Timestamp("2025-01-01T00:15:00Z")].iloc[0]
        self.assertTrue(pd.isna(row.m15_close_time))

    def test_m15_state_carries_forward_only_after_close(self):
        out = align_m15_structure_to_m5(self.make_data())
        first = out[out.Datetime < pd.Timestamp("2025-01-01T00:15:00Z")]
        eligible = out[out.Datetime >= pd.Timestamp("2025-01-01T00:15:00Z")]
        self.assertTrue(first.m15_close_time.isna().all())
        self.assertTrue(eligible.m15_close_time.notna().all())


if __name__ == "__main__":
    unittest.main()
