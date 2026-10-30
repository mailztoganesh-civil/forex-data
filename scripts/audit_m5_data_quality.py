#!/usr/bin/env python3
"""Audit Dukascopy BID/ASK M5 files without deleting or changing raw data.

Writes daily_quality_audit.csv and flagged_m5_bars.csv beside the input files.
A row count alone is not treated as proof of a valid trading day.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

PRICE_COLS = [
    "BidOpen", "BidHigh", "BidLow", "BidClose",
    "AskOpen", "AskHigh", "AskLow", "AskClose",
]
VOLUME_COLS = ["BidVolume", "AskVolume"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/EURUSD/5m")
    args = ap.parse_args()
    folder = Path(args.input_dir)
    files = sorted(folder.glob("*.csv"))
    files = [p for p in files if p.name not in {
        "data_quality_report.csv", "daily_quality_audit.csv", "flagged_m5_bars.csv"
    }]
    if not files:
        raise SystemExit(f"No annual CSV files found in {folder}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if not frame.empty:
            frame["SourceFile"] = path.name
            frames.append(frame)
    if not frames:
        raise SystemExit("Annual CSV files contain no rows")

    df = pd.concat(frames, ignore_index=True)
    required = ["Datetime", *PRICE_COLS, *VOLUME_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    for col in PRICE_COLS + VOLUME_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    df["Date"] = df["Datetime"].dt.strftime("%Y-%m-%d")
    df["Weekday"] = df["Datetime"].dt.day_name()
    df["VolumeSum"] = df[VOLUME_COLS].fillna(0).sum(axis=1)
    df["PriceRange"] = (
        (df["BidHigh"] - df["BidLow"]).abs()
        + (df["AskHigh"] - df["AskLow"]).abs()
    )
    df["ZeroVolume"] = df["VolumeSum"] <= 0
    df["FlatPrice"] = df["PriceRange"] <= 1e-12
    df["SuspiciousFlatZeroVolume"] = df["ZeroVolume"] & df["FlatPrice"]
    df["TimestampAligned"] = (
        (df["Datetime"].dt.minute % 5 == 0)
        & (df["Datetime"].dt.second == 0)
    )
    df["OHLCValid"] = (
        (df["BidHigh"] >= df[["BidOpen", "BidClose", "BidLow"]].max(axis=1))
        & (df["BidLow"] <= df[["BidOpen", "BidClose", "BidHigh"]].min(axis=1))
        & (df["AskHigh"] >= df[["AskOpen", "AskClose", "AskLow"]].max(axis=1))
        & (df["AskLow"] <= df[["AskOpen", "AskClose", "AskHigh"]].min(axis=1))
        & (df[PRICE_COLS] > 0).all(axis=1)
    )

    duplicate_mask = df.duplicated("Datetime", keep=False)
    df["DuplicateTimestamp"] = duplicate_mask
    df["GapFromPreviousMinutes"] = df["Datetime"].diff().dt.total_seconds() / 60

    daily = []
    for date, g in df.groupby("Date", sort=True):
        weekday = g["Weekday"].iloc[0]
        rows = len(g)
        active = int((~g["ZeroVolume"] & ~g["FlatPrice"]).sum())
        zero_flat = int(g["SuspiciousFlatZeroVolume"].sum())
        duplicates = int(g["DuplicateTimestamp"].sum())
        bad_ohlc = int((~g["OHLCValid"]).sum())
        misaligned = int((~g["TimestampAligned"]).sum())
        gaps = int((g["GapFromPreviousMinutes"].fillna(5) > 5).sum())
        day_range = float(g["BidHigh"].max() - g["BidLow"].min())
        if active == 0 and weekday in ("Saturday", "Sunday"):
            status = "weekend_no_activity"
        elif active == 0:
            status = "no_activity_weekday_check_holiday_or_feed"
        elif rows < 200:
            status = "partial_day_or_missing_bars"
        elif zero_flat > 0 or duplicates or bad_ohlc or misaligned:
            status = "active_day_with_quality_warnings"
        else:
            status = "active_day_review_gaps"
        daily.append({
            "date": date, "weekday_utc": weekday, "m5_rows": rows,
            "active_nonflat_positive_volume_bars": active,
            "flat_zero_volume_bars": zero_flat,
            "zero_volume_bars": int(g["ZeroVolume"].sum()),
            "duplicate_timestamp_rows": duplicates,
            "timestamp_misaligned_rows": misaligned,
            "invalid_ohlc_rows": bad_ohlc,
            "gaps_over_5_minutes": gaps,
            "bid_day_range": day_range,
            "status": status,
        })

    audit = pd.DataFrame(daily)
    audit.to_csv(folder / "daily_quality_audit.csv", index=False)
    flags = df[
        df["SuspiciousFlatZeroVolume"] | df["DuplicateTimestamp"]
        | ~df["TimestampAligned"] | ~df["OHLCValid"]
    ].copy()
    flags.to_csv(folder / "flagged_m5_bars.csv", index=False)

    print(f"Audited {len(df)} rows across {len(audit)} UTC dates")
    print(audit.to_string(index=False))
    print(f"Wrote {folder / 'daily_quality_audit.csv'}")
    print(f"Wrote {folder / 'flagged_m5_bars.csv'}")


if __name__ == "__main__":
    main()
