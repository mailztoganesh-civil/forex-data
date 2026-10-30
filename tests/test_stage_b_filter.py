import unittest
from scripts.backtest_phantom_stage_a import structure_allows


class StructureFilterTests(unittest.TestCase):
    def test_long_requires_confirmed_bullish_m15_state(self):
        self.assertTrue(structure_allows("long", 1))
        self.assertFalse(structure_allows("long", 0))
        self.assertFalse(structure_allows("long", -1))

    def test_short_requires_confirmed_bearish_m15_state(self):
        self.assertTrue(structure_allows("short", -1))
        self.assertFalse(structure_allows("short", 0))
        self.assertFalse(structure_allows("short", 1))

    def test_unknown_side_is_rejected(self):
        self.assertFalse(structure_allows(None, 1))


if __name__ == "__main__":
    unittest.main()
