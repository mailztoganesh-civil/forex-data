#!/usr/bin/env python3
"""Download Dukascopy EURUSD M1 BID/ASK candles and aggregate them to M5.

This is a data pipeline, not a trading strategy. Spread fields derived from
minute OHLC are approximations, not tick-level execution prices.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import lzma
import struct
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

BASE = "https://datafeed.dukascopy.com/datafeed"
PAIR_POINT = {"EURUSD": 1e-5}
RECORD = struct.Struct(">5If")  # seconds from UTC day start, O, C, L, H, volume


def fetch_day(pair: str, day: dt.date, side: str, retries: int = 4) -> pd.DataFrame:
    url = (
        f"{BASE}/{pair}/{day.year}/{day.month - 1:02d}/{day.day:02d}/"
        f"{side}_candles_min_1.bi5"
    )
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=30) as response:
                raw = response.read()
            if not raw:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            payload = lzma.decompress(raw)
            rows = []
            for offset in range(0, len(payload) - RECORD.size + 1, RECORD.size):
                seconds, op, close, low, high, volume = RECORD.unpack_from(payload, offset)
                stamp = pd.Timestamp(day, tz="UTC") + pd.Timedelta(seconds=seconds)
                point = PAIR_POINT[pair]
                rows.append((stamp, op * point, high * point, low * point, close * point, volume))
            if not rows:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            return pd.DataFrame(
                rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
            ).set_index("Datetime").sort_index()
        except HTTPError as exc:
            if exc.code == 404:
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            last_error = exc
        except (URLError, TimeoutError, lzma.LZMAError, OSError, struct.error) as exc:
            last_error = exc
        time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"Failed {pair} {side} {day}: {last_error}")


def to_m5(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.resample("5min", label="left", closed="left").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])


def read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["Datetime"])
    index = pd.DatetimeIndex(frame.pop("Datetime"))
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    frame.index = index
    return frame.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start", required=True, help="Inclusive UTC date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive UTC date YYYY-MM-DD")
    parser.add_argument("--out", default="data")
    parser.add_argument("--pause", type=float, default=0.15)
    args = parser.parse_args()

    pair = args.pair.upper()
    if pair not in PAIR_POINT:
        raise SystemExit(f"Unsupported pair {pair}; define its point size first.")
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on/after --start")

    outdir = Path(args.out) / pair / "5m"
    outdir.mkdir(parents=True, exist_ok=True)
    ledger_path = outdir / "download_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    cache = {year: read_existing(outdir / f"{year}.csv") for year in sorted({d.year for d in dates})}
    quality = []

    for day in dates:
        key = day.isoformat()
        if ledger.get(key, {}).get("status") == "complete":
            quality.append({"date": key, "status": "already_complete"})
            continue

        bid_raw = fetch_day(pair, day, "BID")
        time.sleep(args.pause)
        ask_raw = fetch_day(pair, day, "ASK")
        time.sleep(args.pause)
        bid, ask = to_m5(bid_raw).add_prefix("Bid"), to_m5(ask_raw).add_prefix("Ask")
        paired = bid.join(ask, how="inner")

        if not paired.empty:
            for field in ("Open", "High", "Low", "Close"):
                paired[f"Spread{field}"] = paired[f"Ask{field}"] - paired[f"Bid{field}"]
                paired[f"Spread{field}Pips"] = paired[f"Spread{field}"] / 1e-4
            new_rows = paired.reset_index()
            new_rows["Datetime"] = pd.to_datetime(new_rows["Datetime"], utc=True)
            year = day.year
            existing = cache[year]
            combined = pd.concat([existing, new_rows.set_index("Datetime")])
            cache[year] = combined[~combined.index.duplicated(keep="last")].sort_index()

        count = len(paired)
        ledger[key] = {"status": "complete", "paired_m5_bars": int(count)}
        quality.append({"date": key, "status": "downloaded", "paired_m5_bars": int(count),
                        "full_day_reference_bars": 288})
        ledger_path.write_text(json.dumps(ledger, indent=2))
        print(f"{key}: {count} paired M5 bars")

    for year, frame in cache.items():
        if not frame.empty:
            frame = frame.sort_index()
            frame.index.name = "Datetime"
            frame.reset_index().to_csv(outdir / f"{year}.csv", index=False,
                                       date_format="%Y-%m-%dT%H:%M:%SZ")
    pd.DataFrame(quality).to_csv(outdir / "data_quality_report.csv", index=False)
    print(f"Saved data under {outdir}")


if __name__ == "__main__":
    main()
