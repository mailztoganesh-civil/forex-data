import unittest
import pandas as pd
from scripts.stage_e_order_block import order_block_retest_events, align_m15_order_blocks_to_m5
from scripts.backtest_phantom_stage_a import order_block_allows


class StageEOrderBlockTests(unittest.TestCase):
    def test_bullish_order_block_later_retest(self):
        o=[10.0,10.1,10.2,9.9,9.9]
        h=[10.2,10.3,10.25,10.7,10.2]
        l=[9.9,10.0,9.8,9.85,9.85]
        c=[10.1,10.2,9.9,10.6,10.15]
        out=order_block_retest_events(o,h,l,c,lookback=2,atr_period=2,
                                       displacement_atr=0.0,max_age=5)
        self.assertFalse(out.loc[3,"bullish_ob_retest"])
        self.assertTrue(out.loc[4,"bullish_ob_retest"])

    def test_bearish_order_block_later_retest(self):
        o=[10.0,9.9,9.8,10.1,10.1]
        h=[10.1,10.0,10.2,10.15,10.15]
        l=[9.8,9.7,9.75,9.3,9.8]
        c=[9.9,9.8,10.1,9.4,9.85]
        out=order_block_retest_events(o,h,l,c,lookback=2,atr_period=2,
                                       displacement_atr=0.0,max_age=5)
        self.assertFalse(out.loc[3,"bearish_ob_retest"])
        self.assertTrue(out.loc[4,"bearish_ob_retest"])

    def test_no_retest_without_directional_close(self):
        o=[10.0,10.1,10.2,9.9,10.0]
        h=[10.2,10.3,10.25,10.7,10.2]
        l=[9.9,10.0,9.8,9.85,9.85]
        c=[10.1,10.2,9.9,10.6,9.95]
        out=order_block_retest_events(o,h,l,c,lookback=2,atr_period=2,
                                       displacement_atr=0.0,max_age=5)
        self.assertFalse(out.loc[4,"bullish_ob_retest"])

    def test_expiry(self):
        o=[10.0,10.1,10.2,9.9,10.0,10.0,9.9]
        h=[10.2,10.3,10.25,10.7,10.2,10.2,10.2]
        l=[9.9,10.0,9.8,9.85,9.85,9.85,9.85]
        c=[10.1,10.2,9.9,10.6,10.0,10.0,10.15]
        out=order_block_retest_events(o,h,l,c,lookback=2,atr_period=2,
                                       displacement_atr=0.0,max_age=1)
        self.assertFalse(out.loc[6,"bullish_ob_retest"])

    def test_directional_filter(self):
        self.assertTrue(order_block_allows("long",True,False))
        self.assertFalse(order_block_allows("long",False,True))
        self.assertTrue(order_block_allows("short",False,True))
        self.assertFalse(order_block_allows("short",True,False))

    def test_m15_event_only_visible_after_close(self):
        times=pd.date_range("2025-01-01T00:00:00Z",periods=9,freq="5min")
        rows=[]
        for i,ts in enumerate(times):
            p=1.1+i*.0001
            rows.append({"Datetime":ts,"BidOpen":p,"BidHigh":p+.0002,
                "BidLow":p-.0002,"BidClose":p,"AskOpen":p+.0001,
                "AskHigh":p+.0003,"AskLow":p-.0001,"AskClose":p+.0001})
        out=align_m15_order_blocks_to_m5(pd.DataFrame(rows))
        self.assertTrue(out.loc[:2,"m15_close_time"].isna().all())
        self.assertEqual(out.loc[3,"m15_close_time"],pd.Timestamp("2025-01-01T00:15:00Z"))


if __name__=="__main__":
    unittest.main()
