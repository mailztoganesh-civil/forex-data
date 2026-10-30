import unittest
from scripts.stage_g_quality_score import directional_confirmation_score


class StageGScoreTests(unittest.TestCase):
    def test_all_bullish_confirmations_score_six(self):
        r = directional_confirmation_score("long", 1, 1, True, False,
                                            True, False, True, False)
        self.assertEqual(r["score"], 6)

    def test_opposing_confirmations_do_not_score(self):
        r = directional_confirmation_score("long", -1, -1, False, True,
                                            False, True, False, True)
        self.assertEqual(r["score"], 0)

    def test_partial_confirmations_are_additive(self):
        r = directional_confirmation_score("short", -1, 0, False, True,
                                            False, False, False, False)
        self.assertEqual(r["score"], 3)
        self.assertEqual(r["h1"], 2)
        self.assertEqual(r["liquidity"], 1)

    def test_invalid_side_scores_zero(self):
        self.assertEqual(directional_confirmation_score("flat")["score"], 0)


if __name__ == "__main__":
    unittest.main()
