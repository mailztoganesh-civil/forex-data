import importlib.util
from argparse import Namespace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "stage_a", Path(__file__).resolve().parents[1] / "scripts" / "backtest_phantom_stage_a.py"
)
stage_a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_a)


class StageATests(unittest.TestCase):
    def test_atr_and_supertrend_direction_are_finite_after_warmup(self):
        close = pd.Series(np.linspace(1.10, 1.12, 100))
        high, low = close + 0.0005, close - 0.0005
        atr = stage_a.wilder_atr(high, low, close, 10)
        direction = stage_a.supertrend_direction(high, low, close, atr, 3.0)
        self.assertTrue(np.isfinite(atr.iloc[-1]))
        self.assertIn(direction.iloc[-1], (-1.0, 1.0))

    def test_empty_signal_data_returns_zero_trade_summary(self):
        n = 150
        c = np.full(n, 1.10)
        df = pd.DataFrame({
            "Datetime": pd.date_range("2021-01-01", periods=n, freq="5min", tz="UTC"),
            "BidOpen": c, "BidHigh": c + 0.0001, "BidLow": c - 0.0001, "BidClose": c,
            "AskOpen": c + 0.0001, "AskHigh": c + 0.0001, "AskLow": c - 0.0001, "AskClose": c + 0.0001,
            "LongSignal": False, "ShortSignal": False, "ATR": 0.001, "MidClose": c + 0.00005,
        })
        args = Namespace(start="2021-01-01", end="2021-01-02", slippage_pips=0.2,
                         commission_pips=0.0, stop_atr=1.5, rr=2.0)
        trades, summary = stage_a.run_backtest(df, args)
        self.assertTrue(trades.empty)
        self.assertEqual(summary["trades"], 0)
        self.assertEqual(summary["net_pips"], 0.0)

    def test_next_bar_entry_uses_executable_quote_side(self):
        # A single long signal on bar 0 enters at bar 1 ask open plus slippage.
        times = pd.date_range("2021-01-01", periods=3, freq="5min", tz="UTC")
        bid = np.array([1.1000, 1.1010, 1.1010])
        ask = bid + 0.0002
        df = pd.DataFrame({
            "Datetime": times,
            "BidOpen": bid, "BidHigh": bid + 0.0002, "BidLow": bid - 0.0002, "BidClose": bid,
            "AskOpen": ask, "AskHigh": ask + 0.0002, "AskLow": ask - 0.0002, "AskClose": ask,
            "LongSignal": [True, False, False], "ShortSignal": False,
            "ATR": 0.002, "MidClose": (bid + ask) / 2,
        })
        args = Namespace(start="2021-01-01", end="2021-01-02", slippage_pips=0.2,
                         commission_pips=0.0, stop_atr=1.5, rr=2.0)
        trades, summary = stage_a.run_backtest(df, args)
        self.assertFalse(trades.empty)
        self.assertEqual(trades.iloc[0].entry_time, times[1])
        self.assertAlmostEqual(trades.iloc[0].entry, ask[1] + 0.00002, places=7)


if __name__ == "__main__":
    unittest.main()
