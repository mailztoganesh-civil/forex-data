import unittest
import pandas as pd
from scripts.stage_d_fvg import fvg_retest_events, align_m15_fvg_to_m5
from scripts.backtest_phantom_stage_a import fvg_allows


class StageDFVGTests(unittest.TestCase):
    def test_bullish_gap_then_later_bullish_retest(self):
        high=[10, 10.5, 12, 11.8]
        low =[9, 10.1, 11, 10.4]
        open_=[9.5, 10.2, 11.5, 10.5]
        close=[9.8, 10.4, 11.8, 11.2]
        out=fvg_retest_events(high,low,close,open_,max_age=5)
        self.assertFalse(out.loc[2,"bullish_fvg_retest"])
        self.assertTrue(out.loc[3,"bullish_fvg_retest"])

    def test_bearish_gap_then_later_bearish_retest(self):
        high=[12, 11.5, 10, 10.6]
        low =[11, 10.8, 9, 9.5]
        open_=[11.5, 11.2, 9.5, 10.5]
        close=[11.2, 11.0, 9.2, 9.8]
        out=fvg_retest_events(high,low,close,open_,max_age=5)
        self.assertFalse(out.loc[2,"bearish_fvg_retest"])
        self.assertTrue(out.loc[3,"bearish_fvg_retest"])

    def test_no_retest_without_directional_rejection(self):
        high=[10,10.5,12,11.8]
        low=[9,10.1,11,10.4]
        open_=[9.5,10.2,11.5,10.5]
        close=[9.8,10.4,11.8,10.6]
        out=fvg_retest_events(high,low,close,open_,max_age=5)
        self.assertFalse(out.loc[3,"bullish_fvg_retest"])

    def test_expired_gap_does_not_trigger(self):
        out=fvg_retest_events([10,10.5,12,11.8,11.8],
                              [9,10.1,11,10.4,10.4],
                              [9.8,10.4,11.8,11.2,11.2],
                              [9.5,10.2,11.5,10.5,10.8],max_age=1)
        self.assertFalse(out.loc[4,"bullish_fvg_retest"])

    def test_directional_filter(self):
        self.assertTrue(fvg_allows("long",True,False))
        self.assertFalse(fvg_allows("long",False,True))
        self.assertTrue(fvg_allows("short",False,True))
        self.assertFalse(fvg_allows("short",True,False))

    def test_m15_fvg_published_only_after_close(self):
        times=pd.date_range("2025-01-01T00:00:00Z",periods=9,freq="5min")
        rows=[]
        for i,ts in enumerate(times):
            p=1.1+i*.0001
            rows.append({"Datetime":ts,"BidOpen":p,"BidHigh":p+.0002,
                         "BidLow":p-.0002,"BidClose":p,
                         "AskOpen":p+.0001,"AskHigh":p+.0003,
                         "AskLow":p-.0001,"AskClose":p+.0001})
        out=align_m15_fvg_to_m5(pd.DataFrame(rows))
        self.assertTrue(out.loc[:2,"m15_close_time"].isna().all())
        self.assertEqual(out.loc[3,"m15_close_time"],pd.Timestamp("2025-01-01T00:15:00Z"))


if __name__=="__main__":
    unittest.main()
