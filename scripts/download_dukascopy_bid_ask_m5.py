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
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed"
PAIR_POINT = {"EURUSD": 1e-5}
RECORD = struct.Struct(">5If")  # seconds from UTC day start, O, C, L, H, volume


_local = threading.local()
_stats = {"requests": 0, "retries": 0}
_stats_lock = threading.Lock()


def _session() -> requests.Session:
    # One keep-alive connection per worker thread instead of a new TLS handshake per file.
    if not hasattr(_local, "session"):
        sess = requests.Session()
        sess.headers["User-Agent"] = "Mozilla/5.0"
        _local.session = sess
    return _local.session


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def fetch_day(pair: str, day: dt.date, side: str, retries: int = 8) -> pd.DataFrame:
    url = (
        f"{BASE}/{pair}/{day.year}/{day.month - 1:02d}/{day.day:02d}/"
        f"{side}_candles_min_1.bi5"
    )
    last_error = None
    for attempt in range(retries):
        t0 = time.monotonic()
        wait = min(5 * (2 ** attempt), 60)
        try:
            with _stats_lock:
                _stats["requests"] += 1
            resp = _session().get(url, timeout=30)
            took = time.monotonic() - t0
            if resp.status_code == 404:
                print(f"  {day} {side}: 404 (no data) {took:.1f}s", flush=True)
                return _empty()
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = min(int(retry_after), 120)
                raise _Retry(last_error)
            raw = resp.content
            if not raw:
                return _empty()
            payload = lzma.decompress(raw)
            rows = []
            point = PAIR_POINT[pair]
            day_start = pd.Timestamp(day, tz="UTC")
            for offset in range(0, len(payload) - RECORD.size + 1, RECORD.size):
                seconds, op, close, low, high, volume = RECORD.unpack_from(payload, offset)
                stamp = day_start + pd.Timedelta(seconds=seconds)
                rows.append((stamp, op * point, high * point, low * point, close * point, volume))
            if attempt:
                print(f"  {day} {side}: OK after {attempt} retr{'y' if attempt == 1 else 'ies'}", flush=True)
            if not rows:
                return _empty()
            return pd.DataFrame(
                rows, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
            ).set_index("Datetime").sort_index()
        except _Retry:
            pass
        except (requests.RequestException, lzma.LZMAError, OSError, struct.error) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        took = time.monotonic() - t0
        with _stats_lock:
            _stats["retries"] += 1
        if attempt < retries - 1:
            print(f"  {day} {side}: {last_error} after {took:.1f}s, "
                  f"retry {attempt + 1}/{retries - 1} in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed {pair} {side} {day}: {last_error}")


class _Retry(Exception):
    pass


def fetch_pair(pair: str, day: dt.date, pause: float):
    """Fetch BID and ASK for one day; return (day, bid, ask, error)."""
    try:
        bid = fetch_day(pair, day, "BID")
        time.sleep(pause)
        ask = fetch_day(pair, day, "ASK")
        time.sleep(pause)
        return day, bid, ask, None
    except RuntimeError as exc:
        return day, None, None, exc


def to_m5(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.resample("5min", label="left", closed="left").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])


def year_file(outdir: Path, year: int) -> Path:
    return outdir / f"{year}.csv.gz"


def read_existing(outdir: Path, year: int) -> pd.DataFrame:
    path = year_file(outdir, year)
    if not path.exists():
        legacy = outdir / f"{year}.csv"   # older uncompressed output
        if not legacy.exists():
            return pd.DataFrame()
        path = legacy
    frame = pd.read_csv(path, parse_dates=["Datetime"])
    index = pd.DatetimeIndex(frame.pop("Datetime"))
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    frame.index = index
    return frame.sort_index()


def save_all(cache: dict, outdir: Path, quality: list) -> None:
    for year, frame in cache.items():
        if not frame.empty:
            frame = frame.sort_index()
            frame.index.name = "Datetime"
            frame.reset_index().to_csv(year_file(outdir, year), index=False,
                                       date_format="%Y-%m-%dT%H:%M:%SZ")
    pd.DataFrame(quality).to_csv(outdir / "data_quality_report.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start", required=True, help="Inclusive UTC date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive UTC date YYYY-MM-DD")
    parser.add_argument("--out", default="data")
    parser.add_argument("--subdir", default="5m_bidask",
                        help="Folder under <out>/<PAIR>/ (kept apart from HistData's 5m/)")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Seconds each worker waits between requests")
    parser.add_argument("--workers", type=int, default=3,
                        help="Days downloaded in parallel (lower it if 503s increase)")
    args = parser.parse_args()

    pair = args.pair.upper()
    if pair not in PAIR_POINT:
        raise SystemExit(f"Unsupported pair {pair}; define its point size first.")
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on/after --start")

    outdir = Path(args.out) / pair / args.subdir
    outdir.mkdir(parents=True, exist_ok=True)
    ledger_path = outdir / "download_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    cache = {year: read_existing(outdir, year) for year in sorted({d.year for d in dates})}
    quality = []
    failed: list[str] = []

    pending = [d for d in dates if ledger.get(d.isoformat(), {}).get("status") != "complete"]
    for d in dates:
        if d not in pending:
            quality.append({"date": d.isoformat(), "status": "already_complete"})
    print(f"{len(dates)} day(s) requested, {len(pending)} to download, "
          f"workers={args.workers}, pause={args.pause}s", flush=True)
    t_start = time.monotonic()

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    results = pool.map(lambda d: fetch_pair(pair, d, args.pause), pending)
    for day, bid_raw, ask_raw, exc in results:  # yields in date order
        key = day.isoformat()
        if exc is not None:
            # Skip this day instead of aborting the whole backfill; rerun later to retry it.
            ledger[key] = {"status": "failed", "error": str(exc)}
            quality.append({"date": key, "status": "failed", "error": str(exc)})
            failed.append(key)
            ledger_path.write_text(json.dumps(ledger, indent=2))
            print(f"{key}: FAILED ({exc})", flush=True)
            continue
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
        elapsed = time.monotonic() - t_start
        print(f"{key}: {count} paired M5 bars  [{elapsed / 60:.1f} min elapsed]", flush=True)
        if (day + dt.timedelta(days=1)).day == 1:
            save_all(cache, outdir, quality)  # checkpoint at each month end

    pool.shutdown()
    save_all(cache, outdir, quality)
    elapsed = time.monotonic() - t_start
    print(f"Saved data under {outdir}")
    print(f"Done in {elapsed / 60:.1f} min: {_stats['requests']} requests, "
          f"{_stats['retries']} failed attempts retried", flush=True)
    if failed:
        print(f"WARNING: {len(failed)} day(s) failed and were skipped: {', '.join(failed)}")
        if len(failed) == len(dates):
            raise SystemExit("Every requested day failed; nothing downloaded.")


if __name__ == "__main__":
    main()
