#!/usr/bin/env python3
"""Merge BID/ASK M5 chunk outputs (one per parallel backfill job) into the repo copy.

Usage:
  python scripts/merge_bidask_chunks.py --chunks chunks --dest data/EURUSD/5m_bidask

Each chunk folder holds YYYY.csv.gz files, download_ledger.json and
data_quality_report.csv as written by download_dukascopy_bid_ask_m5.py.
Rows are combined by timestamp (later chunks win on exact duplicates), ledgers
are unioned with "complete" taking priority over "failed".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Folder containing chunk folders")
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    sources = [dest] + sorted(p.parent for p in Path(args.chunks).rglob("download_ledger.json"))
    print("Sources:", *[str(s) for s in sources], sep="\n  ")

    # Price files, by year
    by_year: dict[str, list[pd.DataFrame]] = {}
    for src in sources:
        for f in sorted(src.glob("20[0-9][0-9].csv*")):
            by_year.setdefault(f.name.split(".")[0], []).append(pd.read_csv(f))
    for year, frames in sorted(by_year.items()):
        data = pd.concat(frames, ignore_index=True)
        data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
        data = data.drop_duplicates("Datetime", keep="last").sort_values("Datetime")
        out = dest / f"{year}.csv.gz"
        data.to_csv(out, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
        legacy = dest / f"{year}.csv"
        if legacy.exists():
            legacy.unlink()
        print(f"{out}: {len(data):,} bars, {data.Datetime.min()} -> {data.Datetime.max()}")

    # Ledger
    ledger: dict = {}
    for src in sources:
        path = src / "download_ledger.json"
        if not path.exists():
            continue
        for day, entry in json.loads(path.read_text()).items():
            if ledger.get(day, {}).get("status") == "complete" and entry.get("status") != "complete":
                continue
            ledger[day] = entry
    (dest / "download_ledger.json").write_text(json.dumps(dict(sorted(ledger.items())), indent=2))
    done = sum(1 for e in ledger.values() if e.get("status") == "complete")
    failed = sorted(d for d, e in ledger.items() if e.get("status") == "failed")
    print(f"Ledger: {done} complete day(s), {len(failed)} failed")
    if failed:
        print("Failed days (rerun the backfill to retry):", ", ".join(failed))

    # Quality report: keep the most informative row per date
    reports = [pd.read_csv(s / "data_quality_report.csv") for s in sources
               if (s / "data_quality_report.csv").exists()]
    if reports:
        q = pd.concat(reports, ignore_index=True)
        q["_rank"] = q["status"].map({"downloaded": 0, "failed": 1}).fillna(2)
        q = q.sort_values(["date", "_rank"]).drop_duplicates("date").drop(columns="_rank")
        q.to_csv(dest / "data_quality_report.csv", index=False)


if __name__ == "__main__":
    main()
