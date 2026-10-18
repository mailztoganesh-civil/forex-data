"""
Dukascopy 5-minute OHLC backfill (true OHLC built from Dukascopy 1-min BID candles).

Usage:
  python fetch_dukascopy.py              # last 365 days, EURUSD
  python fetch_dukascopy.py 730          # last 730 days
  python fetch_dukascopy.py 365 GBPUSD   # other pair

Output (UTC timestamps, merged into existing files, safe to re-run):
  data/<PAIR>/5m/YYYY.csv   Datetime,Open,High,Low,Close,Volume
"""

import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 365
PAIR = sys.argv[2].upper() if len(sys.argv) > 2 else "EURUSD"
POINT = 1e-3 if PAIR.endswith("JPY") else 1e-5
URL = "https://datafeed.dukascopy.com/datafeed/{p}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
OUT = Path("data") / PAIR / "5m"
REC = struct.Struct(">5if")  # time offset s, open, close, low, high (points), volume


def fetch_day(day: date):
    url = URL.format(p=PAIR, y=day.year, m=day.month - 1, d=day.day)  # month is 0-based
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            time.sleep(2 + attempt * 3)
        except Exception:
            time.sleep(2 + attempt * 3)
    else:
        print(f"  FAILED {day}")
        return []
    if not raw:
        return []
    data = lzma.decompress(raw)
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rows = []
    for t, o, c, lo, hi, v in REC.iter_unpack(data):
        if v <= 0:  # weekend / no-trade filler bars
            continue
        rows.append((base + timedelta(seconds=t), o * POINT, hi * POINT, lo * POINT, c * POINT, v))
    return rows


def main():
    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=i) for i in range(DAYS, 0, -1)]  # excludes today (incomplete)
    days = [d for d in days if d.weekday() != 5]  # Saturdays have no data
    print(f"{PAIR}: fetching {len(days)} days {days[0]} -> {days[-1]}")
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(fetch_day, days):
            rows.extend(r)
    if not rows:
        sys.exit("No data downloaded")
    m1 = pd.DataFrame(rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]).set_index("Datetime").sort_index()
    bad = ((m1.Low > m1[["Open", "Close"]].min(axis=1)) | (m1.High < m1[["Open", "Close"]].max(axis=1))).mean()
    print(f"1-min bars: {len(m1)}, inconsistent OHLC: {bad:.2%}")
    m5 = m1.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    OUT.mkdir(parents=True, exist_ok=True)
    for year, part in m5.groupby(m5.index.year):
        path = OUT / f"{year}.csv"
        part = part.reset_index()
        if path.exists():
            old = pd.read_csv(path, parse_dates=["Datetime"])
            if old["Datetime"].dt.tz is None:
                old["Datetime"] = old["Datetime"].dt.tz_localize("UTC")
            part = pd.concat([old, part]).drop_duplicates("Datetime", keep="last").sort_values("Datetime")
        part[["Open", "High", "Low", "Close"]] = part[["Open", "High", "Low", "Close"]].round(6)
        part["Volume"] = part["Volume"].round(2)
        part.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"  {path}: {len(part)} rows")


if __name__ == "__main__":
    main()
