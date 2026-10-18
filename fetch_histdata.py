"""
HistData.com 5-minute OHLC backfill (true OHLC built from HistData free 1-min ASCII bars).

Usage:
  python -u fetch_histdata.py              # 2025 -> now, EURUSD
  python -u fetch_histdata.py 2023         # 2023 -> now
  python -u fetch_histdata.py 2024 GBPUSD  # other pair

HistData times are EST without daylight saving (fixed UTC-5); converted to UTC here.
Past years come as one yearly file, the current year as monthly files.

Output (UTC timestamps, merged into existing files, safe to re-run):
  data/<PAIR>/5m/YYYY.csv              Datetime,Open,High,Low,Close,Volume
  data/<PAIR>/1m_histdata/YYYY-MM.csv  same columns, 1-minute (build any timeframe from these)
"""

import http.cookiejar
import io
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

START_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
PAIR = sys.argv[2].upper() if len(sys.argv) > 2 else "EURUSD"
PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{p}/{period}"
GET = "https://www.histdata.com/get.php"
OUT = Path("data") / PAIR / "5m"
OUT1 = Path("data") / PAIR / "1m_histdata"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def log(*a):
    print(*a, flush=True)


def fetch(year, month=None):
    """Download one yearly (month=None) or monthly zip; return 1-min DataFrame or None."""
    period = f"{year}" if month is None else f"{year}/{month}"
    label = f"{year}" if month is None else f"{year}-{month:02d}"
    page = PAGE.format(p=PAIR.lower(), period=period)
    html = opener.open(urllib.request.Request(page, headers={"User-Agent": UA}), timeout=30).read().decode("utf-8", "ignore")
    m = re.search(r'id="tk"\s+value="([^"]+)"', html) or re.search(r'name="tk"\s+value="([^"]+)"', html)
    if not m:
        log(f"  {label}: no download on page (not published yet?)")
        return None
    form = {"tk": m.group(1), "date": str(year),
            "datemonth": f"{year}" if month is None else f"{year}{month:02d}",
            "platform": "ASCII", "timeframe": "M1", "fxpair": PAIR}
    req = urllib.request.Request(GET, data=urllib.parse.urlencode(form).encode(),
                                 headers={"User-Agent": UA, "Referer": page,
                                          "Content-Type": "application/x-www-form-urlencoded"})
    raw = opener.open(req, timeout=120).read()
    if not raw.startswith(b"PK"):
        log(f"  {label}: response is not a zip ({len(raw)} bytes)")
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        txt = z.read(name).decode()
    df = pd.read_csv(io.StringIO(txt), sep=";", header=None,
                     names=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)  # EST -> UTC
    log(f"  {label}: {len(df)} bars")
    return df.set_index("Datetime")


def save_1m(m1):
    OUT1.mkdir(parents=True, exist_ok=True)
    for (y, mo), part in m1.groupby([m1.index.year, m1.index.month]):
        path = OUT1 / f"{y}-{mo:02d}.csv"
        part = part.reset_index()
        if path.exists():
            old = pd.read_csv(path, parse_dates=["Datetime"])
            part = pd.concat([old, part]).drop_duplicates("Datetime", keep="last").sort_values("Datetime")
        part.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    log(f"  1-min files saved to {OUT1}")


def save(m1):
    m1 = m1[~m1.index.duplicated(keep="last")].sort_index()
    save_1m(m1)
    m5 = m1.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    OUT.mkdir(parents=True, exist_ok=True)
    for year, part in m5.groupby(m5.index.year):
        path = OUT / f"{year}.csv"
        part = part.reset_index()
        if path.exists():
            old = pd.read_csv(path, parse_dates=["Datetime"])
            part = pd.concat([old, part]).drop_duplicates("Datetime", keep="last").sort_values("Datetime")
        part[["Open", "High", "Low", "Close"]] = part[["Open", "High", "Low", "Close"]].round(6)
        part.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        log(f"  {path}: {len(part)} rows, {part.Datetime.min()} -> {part.Datetime.max()}")


def main():
    now = datetime.now(timezone.utc)
    jobs = [(y, None) for y in range(START_YEAR, now.year)] + [(now.year, mo) for mo in range(1, now.month + 1)]
    log(f"{PAIR}: {len(jobs)} downloads from {START_YEAR}")
    frames = []
    for i, (y, mo) in enumerate(jobs):
        try:
            df = fetch(y, mo)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if i == 0:
                sys.exit(f"First download failed: {msg} - HistData not reachable from GitHub")
            log(f"  {y}-{mo}: FAILED {msg}")
            continue
        if df is not None:
            frames.append(df)
    if not frames:
        sys.exit("No data downloaded")
    save(pd.concat(frames))


if __name__ == "__main__":
    main()
