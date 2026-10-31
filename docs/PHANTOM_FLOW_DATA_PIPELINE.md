# Phantom Flow research — EURUSD data pipeline V1

This workflow downloads paired Dukascopy EURUSD one-minute BID and ASK candles
and aggregates them to five-minute bars. It is a data pipeline only: it does
not implement Phantom Flow signals or report trading performance.

## Limitations

- Spread OHLC fields are derived from corresponding ASK and BID OHLC values.
  They are approximations, not tick-level spread or execution reconstruction.
- M5 bars are aggregated from available minute records; partial five-minute
  groups are not yet rejected. The audit therefore exports a conservative
  `tradable_m5_bars.csv` candidate file, excluding flat/zero-volume bars,
  invalid OHLC, misaligned or duplicate timestamps, and crossed BID/ASK OHLC.
  This filter does not prove every surviving bar is complete or executable;
  inspect the audit and validate the aggregation before performance claims.
- Indicative Dukascopy data is not necessarily identical to a broker's feed.
  Commission, latency, slippage, and stop/limit intrabar ordering need explicit
  modelling in a later backtester.
- The date ledger marks a day complete even if the feed returns no paired bars.
  Inspect the quality report and remove that date from
  `data/EURUSD/5m/download_ledger.json` to retry it.
- `tradable_m5_bars.csv` is a derived research input; annual raw CSV files are
  preserved unchanged. The daily audit distinguishes clean active days from
  days with actual intraday gaps; UTC date boundaries are not counted as gaps.
- BID/ASK data lives in `data/EURUSD/5m_bidask/YYYY.csv.gz`. `data/EURUSD/5m/`
  holds HistData single-price (BID) bars from 2023 and is NOT used by the
  Phantom Flow backtests: they need real BID and ASK OHLC, and no BID/ASK
  values are synthesised from single-price data.

## Run

- **Full history, committed to the repo:** Actions → **EURUSD BID-ASK M5 full
  backfill (commit to repo)** → Run workflow (defaults: 2021-01-01 to
  yesterday, 3-month chunks in parallel). It merges the chunks, audits them,
  commits `data/EURUSD/5m_bidask/`, then starts the A-G comparison. Rerun it to
  retry failed days; completed days are skipped.
- **Quick inspection only (artifact, not committed):** Actions → **EURUSD
  BID-ASK M5 Data Backfill** with a small range such as 2021-01-01 to
  2021-01-07.

## Research plan after data validation

Test each engine separately before combining:
A) Phantom ATR Shift + MA oscillator; B) BOS/CHoCH; C) sweep/reclaim;
D) FVG retest; E) OB retest; F) Phantom + structure; G) combined quality score.

Use chronological in-sample 2021–2024 and out-of-sample 2025–latest complete
2026 data. No lookahead; confirmed H1/M15 bars; report trade count, profit
factor, expectancy, drawdown, MFE/MAE, and monthly/session breakdowns.
