#!/usr/bin/env python3
"""Audit Dukascopy BID/ASK M5 files and export a conservative tradable-bar set.

Raw annual CSVs are never modified. The audit writes daily_quality_audit.csv,
flagged_m5_bars.csv, and tradable_m5_bars.csv beside the input files.
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
GENERATED_FILES = {
    "data_quality_report.csv", "daily_quality_audit.csv",
    "flagged_m5_bars.csv", "tradable_m5_bars.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/EURUSD/5m")
    args = ap.parse_args()
    folder = Path(args.input_dir)
    files = sorted(p for p in folder.glob("*.csv") if p.name not in GENERATED_FILES)
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
    # Invalid timestamps are retained in a separate count, not silently treated as bars.
    invalid_timestamp_rows = int(df["Datetime"].isna().sum())
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
        & (df["Datetime"].dt.microsecond == 0)
    )
    df["OHLCValid"] = (
        (df["BidHigh"] >= df[["BidOpen", "BidClose", "BidLow"]].max(axis=1))
        & (df["BidLow"] <= df[["BidOpen", "BidClose", "BidHigh"]].min(axis=1))
        & (df["AskHigh"] >= df[["AskOpen", "AskClose", "AskLow"]].max(axis=1))
        & (df["AskLow"] <= df[["AskOpen", "AskClose", "AskHigh"]].min(axis=1))
        & (df[PRICE_COLS] > 0).all(axis=1)
        & (df[VOLUME_COLS] >= 0).all(axis=1)
    )
    # BID/ASK candle values are not tick-synchronised, but crossed quoted prices
    # at matching OHLC fields are still suspicious and excluded conservatively.
    df["SpreadValid"] = (
        (df["AskOpen"] >= df["BidOpen"])
        & (df["AskHigh"] >= df["BidHigh"])
        & (df["AskLow"] >= df["BidLow"])
        & (df["AskClose"] >= df["BidClose"])
    )

    df["DuplicateTimestamp"] = df.duplicated("Datetime", keep=False)
    df["GapFromPreviousMinutes"] = df["Datetime"].diff().dt.total_seconds() / 60
    df["Tradable"] = (
        ~df["ZeroVolume"] & ~df["FlatPrice"] & df["OHLCValid"]
        & df["TimestampAligned"] & ~df["DuplicateTimestamp"] & df["SpreadValid"]
    )

    daily = []
    for date, g in df.groupby("Date", sort=True):
        weekday = g["Weekday"].iloc[0]
        rows = len(g)
        active = int((~g["ZeroVolume"] & ~g["FlatPrice"]).sum())
        zero_flat = int(g["SuspiciousFlatZeroVolume"].sum())
        duplicates = int(g["DuplicateTimestamp"].sum())
        bad_ohlc = int((~g["OHLCValid"]).sum())
        bad_spread = int((~g["SpreadValid"]).sum())
        misaligned = int((~g["TimestampAligned"]).sum())
        gaps = int((g["GapFromPreviousMinutes"].fillna(5) > 5).sum())
        tradable = int(g["Tradable"].sum())
        day_range = float(g["BidHigh"].max() - g["BidLow"].min())
        if active == 0 and weekday in ("Saturday", "Sunday"):
            status = "weekend_no_activity"
        elif active == 0:
            status = "no_activity_weekday_check_holiday_or_feed"
        elif rows < 200:
            status = "partial_day_or_missing_bars"
        elif duplicates or bad_ohlc or bad_spread or misaligned:
            status = "active_day_with_quality_warnings"
        elif gaps > 0:
            status = "active_day_review_gaps"
        else:
            status = "active_day_clean"
        daily.append({
            "date": date, "weekday_utc": weekday, "m5_rows": rows,
            "active_nonflat_positive_volume_bars": active,
            "tradable_bars": tradable,
            "flat_zero_volume_bars": zero_flat,
            "zero_volume_bars": int(g["ZeroVolume"].sum()),
            "duplicate_timestamp_rows": duplicates,
            "timestamp_misaligned_rows": misaligned,
            "invalid_ohlc_rows": bad_ohlc,
            "crossed_or_invalid_spread_rows": bad_spread,
            "gaps_over_5_minutes": gaps,
            "bid_day_range": day_range,
            "status": status,
        })

    audit = pd.DataFrame(daily)
    audit.to_csv(folder / "daily_quality_audit.csv", index=False)
    flags = df[
        df["SuspiciousFlatZeroVolume"] | df["DuplicateTimestamp"]
        | ~df["TimestampAligned"] | ~df["OHLCValid"] | ~df["SpreadValid"]
    ].copy()
    flags.to_csv(folder / "flagged_m5_bars.csv", index=False)

    # Separate clean execution candidate file; raw yearly files remain untouched.
    # Drop audit-only helper columns so the export retains a straightforward schema.
    helper_cols = [
        "Date", "Weekday", "VolumeSum", "PriceRange", "ZeroVolume",
        "FlatPrice", "SuspiciousFlatZeroVolume", "TimestampAligned",
        "OHLCValid", "SpreadValid", "DuplicateTimestamp",
        "GapFromPreviousMinutes", "Tradable",
    ]
    tradable = df.loc[df["Tradable"]].drop(columns=helper_cols, errors="ignore")
    tradable.to_csv(folder / "tradable_m5_bars.csv", index=False)

    print(f"Audited {len(df)} timestamp-valid rows across {len(audit)} UTC dates")
    print(f"Rows with invalid timestamps: {invalid_timestamp_rows}")
    print(audit.to_string(index=False))
    print(f"Tradable bars exported: {len(tradable)}")
    print(f"Wrote {folder / 'daily_quality_audit.csv'}")
    print(f"Wrote {folder / 'flagged_m5_bars.csv'}")
    print(f"Wrote {folder / 'tradable_m5_bars.csv'}")


if __name__ == "__main__":
    main()
