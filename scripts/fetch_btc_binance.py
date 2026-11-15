"""
BTC 5-minute history from Binance's public data archive (data.binance.vision), BTCUSDT spot.

Yahoo keeps only ~60 days of 5-min crypto data, so this backfills years of it.
Monthly zips for finished months, daily zips for the current month.

Usage:  python scripts/fetch_btc_binance.py [START_YYYY-MM]   (default 2021-01)
Output: data/BTCUSD/5m/YYYY.csv.gz   Datetime,Open,High,Low,Close,Volume  (UTC, bar open time)
Safe to re-run: merges into existing files; finished months already on disk are skipped.
"""

import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/spot/{kind}/klines/BTCUSDT/5m/BTCUSDT-5m-{tag}.zip"
OUT = Path("data/BTCUSD/5m")
COLS = ["Datetime", "Open", "High", "Low", "Close", "Volume"]


def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(3)
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"failed: {url}")


def parse(blob):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=range(6))
    if not str(df.iloc[0, 0]).isdigit():          # some files have a header row
        df = df.iloc[1:]
    t = df[0].astype("int64")
    unit = "us" if t.iloc[0] > 1e14 else "ms"       # Binance switched to microseconds in 2025
    out = pd.DataFrame({"Datetime": pd.to_datetime(t, unit=unit, utc=True)})
    for i, c in enumerate(COLS[1:], start=1):
        out[c] = df[i].astype(float)
    return out


def have_month(existing, y, m):
    if existing is None:
        return False
    s = existing[(existing["Datetime"].dt.year == y) & (existing["Datetime"].dt.month == m)]
    return len(s) >= 28 * 288


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "2021-01"
    y, m = map(int, start.split("-"))
    today = datetime.now(timezone.utc).date()
    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}

    def existing(year):
        if year not in frames:
            p = OUT / f"{year}.csv.gz"
            frames[year] = pd.read_csv(p, parse_dates=["Datetime"]) if p.exists() else None
            if frames[year] is not None and frames[year]["Datetime"].dt.tz is None:
                frames[year]["Datetime"] = frames[year]["Datetime"].dt.tz_localize("UTC")
        return frames[year]

    new = []
    while (y, m) < (today.year, today.month):
        if not have_month(existing(y), y, m):
            blob = get(BASE.format(kind="monthly", tag=f"{y}-{m:02d}"))
            if blob:
                new.append(parse(blob))
                print(f"{y}-{m:02d} monthly ok", flush=True)
            else:
                print(f"{y}-{m:02d} monthly missing", flush=True)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    d = date(today.year, today.month, 1)
    while d < today:                                 # current month, day by day
        blob = get(BASE.format(kind="daily", tag=d.isoformat()))
        if blob:
            new.append(parse(blob))
        d += timedelta(days=1)
    print(f"current month: daily files up to {d - timedelta(days=1)}", flush=True)

    if not new:
        print("nothing new")
        return
    allnew = pd.concat(new, ignore_index=True)
    for year, part in allnew.groupby(allnew["Datetime"].dt.year):
        old = existing(year)
        df = pd.concat([old, part], ignore_index=True) if old is not None else part
        df = df.drop_duplicates("Datetime", keep="last").sort_values("Datetime")
        df.to_csv(OUT / f"{year}.csv.gz", index=False, date_format="%Y-%m-%d %H:%M:%S",
                  float_format="%.2f", compression="gzip")
        print(f"{year}: {len(df)} bars, {df['Datetime'].iloc[0]} -> {df['Datetime'].iloc[-1]}")


if __name__ == "__main__":
    main()
