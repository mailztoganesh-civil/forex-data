"""
PDF trade report: Zero Lag Trend bot on BTCUSD daily, last N years.

Same rules as ctrader-zerolag/ZeroLagTrendBot (TrendFlip entries, Band stop, 2R target,
close on opposite flip, 2-loss side block), via scripts/btc_backtest.py.

Usage: python scripts/zerolag_btc_report.py [--years 5] [--cost-pct 0.06] [--out results/btc]
"""

import argparse
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).parent))
import btc_backtest as bt  # noqa: E402

UP, DN = "#1f5fbf", "#d62728"           # blue up-candles, red down-candles
EXIT_COL = {"TP": "#2ca02c", "SL": "#000000", "flip": "#7f7f7f", "end": "#7f7f7f"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def candles(ax, d, width=0.7):
    x = mdates.date2num(d.index.tz_localize(None).to_pydatetime())
    up = d["close"] >= d["open"]
    for col, m in ((UP, up), (DN, ~up)):
        ax.vlines(x[m], d["low"][m], d["high"][m], color=col, linewidth=0.6)
        ax.bar(x[m], (d["close"] - d["open"])[m].abs().clip(lower=d["close"][m] * 1e-4),
               bottom=np.minimum(d["open"], d["close"])[m], width=width, color=col, linewidth=0)
    ax.xaxis_date()


def ts(t):
    return mdates.date2num(pd.Timestamp(t).tz_localize(None))


def marker(ax, n, side, t, price, offset):
    ax.scatter(ts(t), price, marker="^" if side == 1 else "v", s=46, color="black", zorder=5)
    ax.annotate(str(n), (ts(t), price), xytext=(0, -11 if side == 1 else 6), textcoords="offset points",
                ha="center", fontsize=6.5, fontweight="bold", zorder=6)


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--cost-pct", type=float, default=0.06)
    ap.add_argument("--out", default="results/btc")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = Path(tempfile.mkdtemp())          # chart images only live until the PDF is built

    d1 = bt.load("daily.csv")
    zl, vol, trend, _ = bt.zero_lag_lines(d1)
    lines = pd.DataFrame({"zl": zl, "up": zl + vol, "dn": zl - vol}, index=d1.index)
    end = d1.index[-1]
    since = end - pd.DateOffset(years=a.years)
    t = pd.DataFrame(bt.zero_lag(d1, a.cost_pct))
    t = t[t["entry_time"] >= since].reset_index(drop=True)
    t.index += 1
    t["cumR"] = t["R"].cumsum()
    t["days"] = (t["exit_time"] - t["entry_time"]).dt.days
    t.to_csv(out / f"zerolag_btcusd_d1_{a.years}y_trades.csv", index_label="n")

    # ---------------- stats
    r = t["R"].to_numpy()
    cum = np.cumsum(r)
    dd_r = (cum - np.maximum.accumulate(np.maximum(cum, 0))).min()
    streak = mx = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        mx = max(mx, streak)
    def comp(risk):
        eq = np.cumprod(1 + r * risk / 100)
        return (eq[-1] - 1) * 100, ((eq / np.maximum.accumulate(np.maximum(eq, 1))) - 1).min() * 100
    ret1, dd1 = comp(1)
    ret2, dd2 = comp(2)
    pf = r[r > 0].sum() / -r[r < 0].sum()
    by_reason = t.groupby("reason")["R"].agg(["size", "sum"])
    yearly = t.groupby(t["entry_time"].dt.year)["R"].agg(["size", "sum"])
    first_px, last_px = d1.loc[d1.index >= since, "close"].iloc[0], d1["close"].iloc[-1]

    # ---------------- charts
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    ax.step(np.r_[0, np.arange(1, len(cum) + 1)], np.r_[0, cum], where="post", color="black", linewidth=1.2)
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative R")
    ax.grid(alpha=0.25)
    eq_png = save(fig, img / "equity.png")

    ov_pngs = []
    span = d1[d1.index >= since].index
    cut = span[len(span) // 2]
    for k, (lo_t, hi_t) in enumerate(((span[0], cut), (cut, span[-1]))):
        w = d1[(d1.index >= lo_t) & (d1.index <= hi_t)]
        lw = lines.loc[w.index]
        fig, ax = plt.subplots(figsize=(7.4, 3.9))
        candles(ax, w, width=0.8)
        x = mdates.date2num(w.index.tz_localize(None).to_pydatetime())
        ax.plot(x, lw["zl"], color="black", linewidth=0.8)
        ax.plot(x, lw["up"], color="#888888", linewidth=0.5, linestyle=":")
        ax.plot(x, lw["dn"], color="#888888", linewidth=0.5, linestyle=":")
        for n, row in t.iterrows():
            if lo_t <= row.entry_time <= hi_t:
                marker(ax, n, row.side, row.entry_time, row.entry, 0)
            if lo_t <= row.exit_time <= hi_t:
                ax.scatter(ts(row.exit_time), row.exit, s=16, color=EXIT_COL[row.reason], zorder=5)
        ax.set_xlim(x[0] - 3, x[-1] + 3)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.grid(alpha=0.2)
        ov_pngs.append(save(fig, img / f"overview_{k + 1}.png"))

    trade_pngs = []
    for n, row in t.iterrows():
        i0 = max(d1.index.get_indexer([row.entry_time])[0] - 25, 0)
        i1 = min(d1.index.get_indexer([row.exit_time])[0] + 10, len(d1) - 1)
        w = d1.iloc[i0:i1 + 1]
        lw = lines.iloc[i0:i1 + 1]
        fig, ax = plt.subplots(figsize=(7.4, 2.75))
        candles(ax, w)
        x = mdates.date2num(w.index.tz_localize(None).to_pydatetime())
        ax.plot(x, lw["zl"], color="black", linewidth=0.8)
        ax.plot(x, lw["up"], color="#888888", linewidth=0.5, linestyle=":")
        ax.plot(x, lw["dn"], color="#888888", linewidth=0.5, linestyle=":")
        x0, x1 = ts(row.entry_time), ts(row.exit_time)
        ax.hlines(row.stop, x0, x1, colors=DN, linestyles="--", linewidth=0.9)
        ax.hlines(row.tgt, x0, x1, colors="#2ca02c", linestyles="--", linewidth=0.9)
        ax.hlines(row.entry, x0, x1, colors="black", linestyles=":", linewidth=0.7)
        ax.annotate(f"SL {row.stop:,.0f}", (x0, row.stop), xytext=(2, 2), textcoords="offset points", fontsize=6.5, color=DN)
        ax.annotate(f"TP {row.tgt:,.0f}", (x0, row.tgt), xytext=(2, 2), textcoords="offset points", fontsize=6.5, color="#2ca02c")
        marker(ax, n, row.side, row.entry_time, row.entry, 0)
        ax.scatter(x1, row.exit, s=26, color=EXIT_COL[row.reason], zorder=5)
        lo = min(w["low"].min(), row.stop, row.tgt)
        hi = max(w["high"].max(), row.stop, row.tgt)
        ax.set_ylim(lo - (hi - lo) * 0.04, hi + (hi - lo) * 0.04)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
        ax.grid(alpha=0.2)
        side = "BUY" if row.side == 1 else "SELL"
        ax.set_title(f"#{n}  {side}  {row.entry_time:%d %b %Y} → {row.exit_time:%d %b %Y}  ·  "
                     f"exit: {row.reason}  ·  {row.R:+.2f}R", loc="left", fontsize=8.5, fontweight="bold")
        trade_pngs.append(save(fig, img / f"trade_{n:02d}.png"))

    # ---------------- PDF
    pdf_path = out / f"ZeroLag_BTCUSD_D1_last{a.years}y.pdf"
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=16, spaceAfter=2, alignment=0)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4)
    P = ParagraphStyle("p", parent=ss["Normal"], fontSize=8.5, leading=11)
    S = ParagraphStyle("s", parent=P, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#444444"))
    grid = TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.5),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ])
    story = [
        Paragraph("Zero Lag Trend bot — BTCUSD Daily", H1),
        Paragraph(f"Backtest trade report · {t.entry_time.iloc[0]:%d %b %Y} to {end:%d %b %Y} "
                  f"({a.years} years) · generated {pd.Timestamp.now(tz='UTC'):%d %b %Y}", S),
        Spacer(1, 6),
    ]
    kpi = [
        ["Trades", "Win rate", "Total R", "Profit factor", "Max drawdown", "Worst losing run"],
        [f"{len(r)}", f"{(r > 0).mean() * 100:.0f}%", f"{r.sum():+.1f}R", f"{pf:.2f}",
         f"{dd_r:.1f}R", f"{mx} trades"],
        ["Return @1% risk", "Max DD @1%", "Return @2% risk", "Max DD @2%", "Avg R / trade", "Median hold"],
        [f"{ret1:+.1f}%", f"{dd1:.1f}%", f"{ret2:+.1f}%", f"{dd2:.1f}%", f"{r.mean():+.2f}R",
         f"{t['days'].median():.0f} days"],
    ]
    kt = Table(kpi, colWidths=[30 * mm] * 6)
    kt.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.5),
        ("FONT", (0, 1), (-1, 1), "Helvetica-Bold", 12), ("FONT", (0, 3), (-1, 3), "Helvetica-Bold", 12),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#555555")),
        ("TEXTCOLOR", (0, 2), (-1, 2), colors.HexColor("#555555")),
        ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#999999")),
    ]))
    story += [kt, Spacer(1, 4),
              Paragraph(f"BTC itself went from {first_px:,.0f} to {last_px:,.0f} "
                        f"({(last_px / first_px - 1) * 100:+.0f}%) over the same period.", S),
              Paragraph("Equity curve (R)", H2), Image(str(eq_png), width=180 * mm, height=58 * mm)]

    side_tab = [["", "Trades", "Win %", "Total R"]]
    for sd, nm in ((1, "Buys"), (-1, "Sells")):
        s_ = t[t.side == sd]
        side_tab.append([nm, len(s_), f"{(s_.R > 0).mean() * 100:.0f}%", f"{s_.R.sum():+.1f}"])
    for k, v in by_reason.iterrows():
        side_tab.append([{"TP": "Exit: target", "SL": "Exit: stop", "flip": "Exit: opposite flip",
                          "end": "Exit: still open"}[k], int(v["size"]), "", f"{v['sum']:+.1f}"])
    yr_tab = [["Year", "Trades", "Total R"]] + [[str(y), int(v["size"]), f"{v['sum']:+.1f}"]
                                                for y, v in yearly.iterrows()]
    st, yt = Table(side_tab, colWidths=[34 * mm, 16 * mm, 16 * mm, 18 * mm]), Table(yr_tab, colWidths=[18 * mm, 16 * mm, 18 * mm])
    st.setStyle(grid)
    yt.setStyle(grid)
    story += [Paragraph("Breakdown", H2), Table([[st, yt]], colWidths=[95 * mm, 85 * mm],
                                                style=[("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                       ("LEFTPADDING", (0, 0), (-1, -1), 0)])]
    story += [Paragraph("Rules tested", H2), Paragraph(
        "Same logic as your ZeroLagTrendBot cBot with default settings: Length 70, Band multiplier 1.2, "
        "Entry mode <b>Trend flip</b>, Stop mode <b>Band</b> (stop at the zero-lag line minus the band for buys, plus the band for sells), "
        "target <b>2R</b>, close on opposite flip, max 1 position, loss filter on (2 losses in a row on one side "
        "blocks that side until an opposite trade). Signals on the daily close (00:00 UTC), "
        f"entry at the next day's open. Cost {a.cost_pct}% of price per round trip. "
        "If stop and target are both touched on the same day, the stop is counted. "
        "Data: Yahoo Finance BTC-USD daily (spot, 24/7), not IC Markets CFD prices.", P)]

    story += [PageBreak(), Paragraph("All trades on the chart", H2),
              Paragraph("Black triangle pointing up = buy entry, pointing down = sell entry (numbered as in the "
                        "trade list). Exit dot: green = target, black = stop, grey = opposite flip. "
                        "Solid line = zero-lag EMA, dotted lines = bands.", S), Spacer(1, 3)]
    story += [Image(str(pth), width=180 * mm, height=95 * mm) for pth in ov_pngs]

    rows = [["#", "Side", "Entry date", "Entry", "Stop", "Target", "Exit date", "Exit", "Reason", "Days", "R", "Cum R"]]
    for n, row in t.iterrows():
        rows.append([n, "Buy" if row.side == 1 else "Sell", f"{row.entry_time:%d %b %Y}", f"{row.entry:,.0f}",
                     f"{row.stop:,.0f}", f"{row.tgt:,.0f}", f"{row.exit_time:%d %b %Y}", f"{row.exit:,.0f}",
                     {"TP": "Target", "SL": "Stop", "flip": "Flip", "end": "Open"}[row.reason],
                     row.days, f"{row.R:+.2f}", f"{row.cumR:+.1f}"])
    tt = Table(rows, repeatRows=1, colWidths=[7 * mm, 9 * mm, 19 * mm, 15 * mm, 15 * mm, 15 * mm,
                                              19 * mm, 15 * mm, 12 * mm, 10 * mm, 12 * mm, 13 * mm])
    ts_ = TableStyle(grid.getCommands())
    for k, row in enumerate(rows[1:], start=1):
        ts_.add("TEXTCOLOR", (10, k), (10, k), colors.HexColor("#1a7a1a") if float(row[10]) > 0 else DN)
    tt.setStyle(ts_)
    story += [PageBreak(), Paragraph("Trade list", H2), tt]

    story += [PageBreak(), Paragraph("Trade charts", H2),
              Paragraph("Dashed red = stop, dashed green = target, dotted black = entry price. "
                        "Charts show 25 days before entry and 10 days after exit.", S), Spacer(1, 3)]
    for pth in trade_pngs:
        story.append(KeepTogether([Image(str(pth), width=180 * mm, height=67 * mm), Spacer(1, 4)]))

    def footer(c, doc):
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(15 * mm, 9 * mm, "Zero Lag Trend · BTCUSD D1 · backtest, not a guarantee of future results")
        c.drawRightString(195 * mm, 9 * mm, f"Page {doc.page}")

    SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                      topMargin=13 * mm, bottomMargin=15 * mm, title="Zero Lag Trend bot — BTCUSD D1",
                      author="GR").build(story, onFirstPage=footer, onLaterPages=footer)
    print(pdf_path)


if __name__ == "__main__":
    main()
