import argparse
import unittest
import pandas as pd
from scripts.backtest_phantom_stage_a import run_backtest


class ExecutionSafetyTests(unittest.TestCase):
    def test_no_reentry_at_bar_open_after_intrabar_exit_and_stop_gap(self):
        times = pd.date_range("2025-01-01T00:00:00Z", periods=4, freq="5min")
        rows = []
        for i, ts in enumerate(times):
            bid_open = 1.1000
            bid_low = 1.0995
            bid_high = 1.1005
            signal = False
            if i == 1:
                signal = True  # opens the initial position on this bar
            if i == 2:
                bid_open = 1.0980  # gaps below the prior long stop
                bid_low = 1.0975
                bid_high = 1.1000
            rows.append({
                "Datetime": ts, "BidOpen": bid_open, "BidHigh": bid_high,
                "BidLow": bid_low, "BidClose": bid_open,
                "AskOpen": bid_open + 0.0001, "AskHigh": bid_high + 0.0001,
                "AskLow": bid_low + 0.0001, "AskClose": bid_open + 0.0001,
                "ATR": 0.001, "LongSignal": signal, "ShortSignal": False,
            })
        # Signal on bar 1 is evaluated at bar 2 open, and the position is
        # stopped inside bar 2. It must not be reopened at that same open.
        rows[0]["LongSignal"] = True
        args = argparse.Namespace(start="2025-01-01", end="2025-01-01",
            slippage_pips=0.2, commission_pips=0.0, stop_atr=1.0, rr=2.0,
            structure_filter=False)
        trades, summary = run_backtest(pd.DataFrame(rows), args)
        self.assertEqual(summary["trades"], 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "stop")
        self.assertAlmostEqual(trades.iloc[0]["exit_price"], 1.09798, places=6)


if __name__ == "__main__":
    unittest.main()
