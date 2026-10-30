import unittest
import pandas as pd
from scripts.stage_b_structure import structure_events


class StageBStructureTests(unittest.TestCase):
    def test_pivot_is_only_available_after_right_confirmation(self):
        high = [1, 2, 3, 5, 3, 2, 4]
        low =  [0, 1, 2, 3, 2, 1, 2]
        close=[1, 2, 3, 4, 3, 2, 3]
        x=structure_events(high, low, close, 2, 2)
        self.assertTrue(pd.isna(x.loc[4, "swing_high_confirmed"]))
        self.assertEqual(x.loc[5, "swing_high_confirmed"], 5)

    def test_close_break_marks_bos_once(self):
        high=[1,2,3,2,2,2,2,2]
        low= [0,1,2,1,1,1,1,1]
        close=[1,2,2,2,2,3.1,3.2,3.3]
        x=structure_events(high,low,close,1,1)
        self.assertTrue(x["bos_bull"].any())
        self.assertEqual(int(x["bos_bull"].sum()),1)

    def test_break_against_direction_is_choch(self):
        high=[1,2,3,2,2,2,2,2,2]
        low= [0,1,2,1,0.5,0.8,0.9,0.7,0.6]
        close=[1,2,2,2,1.5,1.6,1.7,0.4,0.3]
        x=structure_events(high,low,close,1,1)
        self.assertTrue(x["choch_bear"].any())


if __name__ == "__main__":
    unittest.main()
