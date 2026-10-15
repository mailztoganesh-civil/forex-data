"""
Forex data collector (Yahoo Finance via yfinance).

Each run fetches what Yahoo allows and MERGES into existing CSVs,
so history keeps growing over time. Safe to run any number of times.

Output (all timestamps UTC):
  data/<PAIR>/daily.csv        full daily history (~2003 onwards)
  data/<PAIR>/1h.csv           hourly, last ~2 years, grows from there
  data/<PAIR>/1m/YYYY-MM.csv   1-minute, last ~30 days, grows from there
  data/summary.json            rows and date range per file
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add more pairs here later, e.g. "GBPUSD", "USDJPY"
PAIRS = ["EURUSD"]

DATA_DIR = Path("data")
COLS = ["Open", "High", "Low", "Close"]


def yahoo_symbol(pair: str) -> str:
    return f"{pair}=X"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Datetime"] + COLS)
    df = df[COLS].copy()
    idx = df.index
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx
    df.index.name = "Datetime"
    df = df.dropna()
    df = df[(df[COLS] > 0).all(axis=1)]
    return df.reset_index()


def merge_save(new: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, parse_dates=["Datetime"])
        if old["Datetime"].dt.tz is None:
            old["Datetime"] = old["Datetime"].dt.tz_localize("UTC")
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    if df.empty:
        return 0
    df = (df.drop_duplicates(subset="Datetime", keep="last")
            .sort_values("Datetime")
            .reset_index(drop=True))
    df[COLS] = df[COLS].round(6)
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return len(df)


def history(symbol, retries=3, **kw) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(auto_adjust=False, **kw)
            return clean(df)
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(5)
    return clean(None)


def fetch_pair(pair: str):
    sym = yahoo_symbol(pair)
    base = DATA_DIR / pair
    now = datetime.now(timezone.utc)

    print(f"{pair}: daily")
    merge_save(history(sym, period="max", interval="1d"), base / "daily.csv")

    print(f"{pair}: 1h")
    start_1h = (now - timedelta(days=729)).date()
    merge_save(history(sym, start=start_1h, interval="1h"), base / "1h.csv")

    # Yahoo: 1m only for last 30 days, max 7 days per request
    print(f"{pair}: 1m")
    chunks = []
    end = now
    stop = now - timedelta(days=29)
    while end > stop:
        start = max(end - timedelta(days=7), stop)
        chunks.append(history(sym, start=start, end=end, interval="1m"))
        end = start
        time.sleep(1)
    m1 = pd.concat([c for c in chunks if not c.empty], ignore_index=True) \
        if any(not c.empty for c in chunks) else clean(None)
    if not m1.empty:
        m1["month"] = m1["Datetime"].dt.strftime("%Y-%m")
        for month, part in m1.groupby("month"):
            merge_save(part.drop(columns="month"), base / "1m" / f"{month}.csv")


def summarize():
    out = {}
    for f in sorted(DATA_DIR.rglob("*.csv")):
        df = pd.read_csv(f, usecols=["Datetime"])
        if df.empty:
            continue
        out[str(f.relative_to(DATA_DIR))] = {
            "rows": len(df),
            "from": df["Datetime"].iloc[0],
            "to": df["Datetime"].iloc[-1],
        }
    out["_updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    (DATA_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    failed = []
    for p in PAIRS:
        try:
            fetch_pair(p)
        except Exception as e:
            print(f"{p} failed: {e}")
            failed.append(p)
    summarize()
    sys.exit(1 if failed and len(failed) == len(PAIRS) else 0)
