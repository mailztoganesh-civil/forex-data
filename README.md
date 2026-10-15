# forex-data

Daily forex data collector (Yahoo Finance) for backtesting. Currently: **EUR/USD**.

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
