#!/usr/bin/env python3
"""Validate downloaded BID/ASK M5 candles and classify inactive dates.

This validator does not equate 288 rows with 288 valid market bars. It checks
timestamps, OHLC consistency, BID/ASK alignment, spreads and price activity.
A flat weekday is reported as NO_ACTIVITY_REVIEW, not silently accepted as
a normal trading day (it may be a holiday or a feed issue).
"""
import argparse
from pathlib import Path
import pandas as pd

PRICE_COLS = [
    "BidOpen", "BidHigh", "BidLow", "BidClose",
    "AskOpen", "AskHigh", "AskLow", "AskClose",
]
REQUIRED = ["Datetime", *PRICE_COLS, "SpreadClosePips"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Annual M5 CSV")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.file)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["Datetime"]).sort_values("Datetime")
    start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    df = df[(df.Datetime >= start) & (df.Datetime < end)].copy()
    df = df.drop_duplicates("Datetime", keep="last")

    rows = []
    fatal = []
    for day in pd.date_range(start.normalize(), (end - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC"):
        d = df[df.Datetime.dt.date == day.date()].copy()
        stamps = pd.DatetimeIndex(d.Datetime)
        expected = pd.date_range(day, periods=288, freq="5min")
        count = len(d)
        missing_slots = len(expected.difference(stamps))
        duplicate_count = int(d.Datetime.duplicated().sum())
        numeric = d[PRICE_COLS + ["SpreadClosePips"]].apply(pd.to_numeric, errors="coerce")
        finite = bool(numeric.notna().all().all()) if count else False

        ohlc_ok = True
        if count:
            for side in ("Bid", "Ask"):
                o, h, l, c = (numeric[f"{side}{x}"] for x in ("Open", "High", "Low", "Close"))
                ohlc_ok &= bool(((h >= pd.concat([o, c, l], axis=1).max(axis=1)) &
                                 (l <= pd.concat([o, c, h], axis=1).min(axis=1))).all())
        active = False
        flat_ratio = 1.0
        spread_ok = False
        if count and finite:
            movement = (numeric.BidHigh != numeric.BidLow) | (numeric.AskHigh != numeric.AskLow)
            active = bool(movement.any())
            flat_ratio = float((~movement).mean())
            spread_ok = bool((numeric.SpreadClosePips > 0).all() and
                             (numeric.SpreadClosePips < 20).all())

        weekday = day.weekday() < 5
        if count == 0:
            status = "NO_DATA"
        elif not finite or not ohlc_ok or not spread_ok or missing_slots or duplicate_count:
            status = "INVALID"
        elif not active:
            status = "NO_ACTIVITY_REVIEW" if weekday else "WEEKEND_NO_ACTIVITY"
        elif count != 288:
            status = "PARTIAL_ACTIVE"
        else:
            status = "VALID_ACTIVE"

        if status == "INVALID":
            fatal.append(str(day.date()))
        rows.append({
            "date_utc": str(day.date()), "weekday": day.strftime("%A"),
            "rows": count, "expected_rows": 288, "missing_5m_slots": missing_slots,
            "duplicate_timestamps": duplicate_count, "active_price_day": active,
            "flat_bar_ratio": round(flat_ratio, 4), "numeric_values_ok": finite,
            "ohlc_consistent": ohlc_ok, "spread_close_plausible": spread_ok,
            "classification": status,
        })

    report = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    print(report.to_string(index=False))
    print("\nClassification counts:")
    print(report.classification.value_counts().to_string())
    print(f"\nQuality report saved: {args.out}")
    if fatal:
        raise SystemExit("Validation failed on structurally invalid date(s): " + ", ".join(fatal))


if __name__ == "__main__":
    main()
