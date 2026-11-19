"""
Dukascopy 5-minute OHLC backfill (true OHLC built from Dukascopy 1-min BID candles).

Usage:
  python -u fetch_dukascopy.py              # last 365 days, EURUSD
  python -u fetch_dukascopy.py 730          # last 730 days
  python -u fetch_dukascopy.py 365 GBPUSD   # other pair

Output (UTC timestamps, merged into existing files, safe to re-run):
  data/<PAIR>/5m/YYYY.csv   Datetime,Open,High,Low,Close,Volume
"""

import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 365
PAIR = sys.argv[2].upper() if len(sys.argv) > 2 else "EURUSD"
import os
POINT = float(os.environ.get("POINT", 1e-3 if (PAIR.endswith("JPY") or "IDX" in PAIR) else 1e-5))
URL = "https://datafeed.dukascopy.com/datafeed/{p}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
OUT = Path("data") / PAIR / "5m"
REC = struct.Struct(">5if")  # time offset s, open, close, low, high (points), volume
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://freeserv.dukascopy.com/"}
TIME_BUDGET = int(os.environ.get("TIME_BUDGET_MIN", 20)) * 60  # stop starting new downloads after 20 min, save what we have
START = time.time()


def log(*a):
    print(*a, flush=True)


def download(day: date, timeout=15):
    url = URL.format(p=PAIR, y=day.year, m=day.month - 1, d=day.day)  # month is 0-based
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse(day: date, raw: bytes):
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


def fetch_day(day: date):
    if time.time() - START > TIME_BUDGET:
        return day, None
    err = ""
    for attempt in range(int(os.environ.get("RETRIES", 3))):
        try:
            return day, parse(day, download(day))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return day, []
            err = f"HTTP {e.code}"
        except Exception as e:
            err = type(e).__name__
        time.sleep(2 + attempt * 3)
    log(f"  FAILED {day}: {err}")
    return day, None


def probe(days):
    """Test one weekday first so a block shows up in seconds, not after 30 min."""
    test = next(d for d in reversed(days) if d.weekday() < 5)
    for attempt in range(6):
        try:
            raw = download(test, timeout=20)
            log(f"Probe {test}: OK, {len(raw)} bytes, {len(parse(test, raw))} bars")
            return
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                log(f"Probe {test}: HTTP {e.code}, waiting 60s"); time.sleep(60); continue
            sys.exit(f"Probe {test}: HTTP {e.code} - Dukascopy is refusing GitHub's servers")
        except Exception as e:
            if attempt < 5:
                log(f"Probe {test}: {type(e).__name__}, waiting 60s"); time.sleep(60); continue
            sys.exit(f"Probe {test}: {type(e).__name__}: {e} - Dukascopy not reachable from GitHub")



def save(rows):
    m1 = pd.DataFrame(rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]).set_index("Datetime").sort_index()
    bad = ((m1.Low > m1[["Open", "Close"]].min(axis=1)) | (m1.High < m1[["Open", "Close"]].max(axis=1))).mean()
    log(f"1-min bars: {len(m1)}, inconsistent OHLC: {bad:.2%}")
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
        log(f"  {path}: {len(part)} rows")


def main():
    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=i) for i in range(DAYS, 0, -1)]  # excludes today (incomplete)
    days = [d for d in days if d.weekday() != 5]  # Saturdays have no data
    log(f"{PAIR}: {len(days)} days {days[0]} -> {days[-1]}")
    probe(days)
    rows, done, failed, retry_days = [], 0, 0, []
    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", 6))) as ex:
        for fut in as_completed([ex.submit(fetch_day, d) for d in days]):
            day, r = fut.result()
            done += 1
            if r is None:
                failed += 1
                retry_days.append(day)
            else:
                rows.extend(r)
            if done % 20 == 0:
                log(f"  {done}/{len(days)} days, {len(rows)} bars, {failed} failed, {time.time() - START:.0f}s")
    if retry_days and time.time() - START < TIME_BUDGET:
        log(f"Retrying {len(retry_days)} failed days one at a time")
        for d in sorted(retry_days):
            time.sleep(1)
            _, r = fetch_day(d)
            if r is not None:
                rows.extend(r); failed -= 1
    log(f"Done: {done - failed} days ok, {failed} failed/skipped")
    if not rows:
        sys.exit("No data downloaded")
    save(rows)


if __name__ == "__main__":
    main()
