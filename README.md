# forex-data

Daily forex data collector (Yahoo Finance) for backtesting. Currently: **EUR/USD** and **BTC/USD** (Yahoo `BTC-USD`, saved as `data/BTCUSD/`).

## Files (UTC timestamps, columns: Datetime, Open, High, Low, Close)

| File | Data |
|---|---|
| `data/EURUSD/daily.csv` | Daily, full Yahoo history (~2003 onward) |
| `data/EURUSD/1h.csv` | Hourly, last ~2 years at start, grows daily |
| `data/EURUSD/1m/YYYY-MM.csv` | 1-minute, last ~30 days at start, grows daily |
| `data/summary.json` | Row counts and date ranges |

Runs daily at 22:17 UTC via GitHub Actions. Run manually: **Actions → Collect forex data → Run workflow**.

## Add more pairs

Edit `PAIRS` in `fetch_forex.py`, e.g. `["EURUSD", "GBPUSD", "USDJPY"]`.

## Notes

- Yahoo FX prices are indicative quotes, not broker prices; volume is not available.
- 1-min history only grows forward. For older 1-min data, use HistData.com or Dukascopy.

## BTC backtest

`scripts/btc_backtest.py` runs Zero Lag Trend, SMA 9/18 pullback and CRT side by side on BTC/USD
(H1 and H4 for the last ~2 years, D1/W1 for the full history), with the same rules as the
ctrader-zerolag bot, the sma-signals worker and the CRT backtest.

Run it: **Actions → BTC backtest → Run workflow**. Results are saved to `results/btc/`
(`summary.md`, `yearly_R.csv`, `trades.csv`) and shown on the run's summary page.

## BTC intraday (Zero Lag M5 / M15 / M30)

`scripts/fetch_btc_binance.py` backfills BTC 5-min bars from Binance's public archive into
`data/BTCUSD/5m/`. `scripts/zl_intraday_research.py` tests the Zero Lag bot settings on
M5/M15/M30/H1 (choices made on the first 3 years, checked on the last 2).
Run: **Actions → BTC intraday Zero Lag research → Run workflow**. Results: `results/btc/intraday/`.

### Deeper research (exits, regime, volume filters)

`scripts/zl_research2.py --holdout` tests extra changes on the M30 + H1 setup. Choices are made on
2021-2026 and confirmed on the untouched 2018-2021 holdout (BTC 5m from 2017 is in `data/BTCUSD/5m`).
Results: `results/btc/research2/`.

## All strategies on currency, gold and BTC

`scripts/all_strategies.py` runs every strategy (Zero Lag M5 defaults, Zero Lag BTC_M30_Trail preset,
Asian breakout, Gold breakout V2, Gold SBS, EMA 34/50, SMA 9/18 pullback, CRT H1/M5) with bot-default
rules on EURUSD, EURJPY, XAUUSD (Dukascopy M5 2016-2026, `data/duka_m5`) and BTCUSD (Binance 2018-2026).
Results: `results/all_strategies/ranking.md`.

