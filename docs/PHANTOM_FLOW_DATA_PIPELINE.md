# Phantom Flow research — EURUSD data pipeline V1

This workflow downloads paired Dukascopy EURUSD one-minute BID and ASK candles
and aggregates them to five-minute bars. It is a data pipeline only: it does
not implement Phantom Flow signals or report trading performance.

## Limitations

- Spread OHLC fields are derived from corresponding ASK and BID OHLC values.
  They are approximations, not tick-level spread or execution reconstruction.
- M5 bars are aggregated from available minute records; partial five-minute
  groups are not yet rejected. Review the quality report before backtesting.
- Indicative Dukascopy data is not necessarily identical to a broker's feed.
  Commission, latency, slippage, and stop/limit intrabar ordering need explicit
  modelling in a later backtester.
- The date ledger marks a day complete even if the feed returns no paired bars.
  Inspect the quality report and remove that date from
  `data/EURUSD/5m/download_ledger.json` to retry it.
- The workflow uploads artifacts with a 30-day retention period; it does not
  commit large market-data files into the repository.

## Run

Open **Actions → EURUSD BID-ASK M5 Data Backfill → Run workflow**.
Start with a small range such as 2021-01-01 through 2021-01-07. Inspect the
artifact's yearly CSV, ledger, and data quality report before larger downloads.

## Research plan after data validation

Test each engine separately before combining:
A) Phantom ATR Shift + MA oscillator; B) BOS/CHoCH; C) sweep/reclaim;
D) FVG retest; E) OB retest; F) Phantom + structure; G) combined quality score.

Use chronological in-sample 2021–2024 and out-of-sample 2025–latest complete
2026 data. No lookahead; confirmed H1/M15 bars; report trade count, profit
factor, expectancy, drawdown, MFE/MAE, and monthly/session breakdowns.
