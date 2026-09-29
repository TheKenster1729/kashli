"""
Plots for data collected by ``btc15m_data.py``.

    plot_market(ticker)      one 15-min window: YES bid/ask/trade price, BTC spot
                             vs. the strike, and per-minute volume, stacked on a
                             shared time axis
    plot_series_overview()   the whole history: calibration of the market-implied
                             probability at several points in the window, volume
                             over time, YES rate over time, and hour-of-day effects

Both return plotly figures; ``save_html`` writes one or more to a single file.

Usage:
    python btc15m_plots.py                            # overview -> kxbtc15m_overview.html
    python btc15m_plots.py KXBTC15M-26SEP232145-45    # one market -> <ticker>.html
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from btc15m_data import load

# Validated categorical slots (light mode) plus recessive chrome.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#9a9893"
GRID, SURFACE = "#e8e7e3", "#fcfcfb"
ET = "America/New_York"  # Kalshi labels these windows in Eastern time

LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK_2),
    title_font=dict(size=16, color=INK),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=70, r=30, t=90, b=50),
)
AXIS = dict(gridcolor=GRID, linecolor=GRID, zeroline=False, ticks="outside", tickcolor=GRID)


def _style(fig):
    fig.update_layout(**LAYOUT)
    fig.update_xaxes(**AXIS)
    fig.update_yaxes(**AXIS)
    return fig


def _candle_times(c):
    """Candle start time (end_period_ts is the end of each 1-minute bucket)."""
    return pd.to_datetime(c.end_period_ts - 60, unit="s", utc=True).dt.tz_convert(ET)


# ── Single market ─────────────────────────────────────────────────────────────

def plot_market(ticker, data=None, spot_padding_min=15):
    data = data or load()
    markets = data["markets"]
    if ticker not in set(markets.ticker):
        raise KeyError(f"{ticker} not in collected markets; re-run btc15m_data.py")
    m = markets.set_index("ticker").loc[ticker]
    c = data["candles"].query("ticker == @ticker").sort_values("end_period_ts")
    t = _candle_times(c)
    open_et, close_et = m.open_time.tz_convert(ET), m.close_time.tz_convert(ET)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.45, 0.35, 0.2],
                        subplot_titles=("YES price ($)", "BTC-USD spot, Coinbase ($)", "Contracts traded / min"))

    # Row 1: book and trades. Bid/ask as a shaded band, trade close as a line.
    fig.add_trace(go.Scatter(x=t, y=c.yes_ask_close, name="YES ask", line=dict(color=BLUE, width=1, shape="hv"),
                             opacity=0.5), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=c.yes_bid_close, name="YES bid", line=dict(color=BLUE, width=1, shape="hv"),
                             fill="tonexty", fillcolor="rgba(42,120,214,0.12)", opacity=0.5), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=c.price_close, name="Last trade (1-min close)",
                             line=dict(color=BLUE, width=2, shape="hv"), mode="lines+markers",
                             marker=dict(size=8, line=dict(color=SURFACE, width=2))), 1, 1)
    trades = data.get("trades")
    if trades is not None:
        tr = trades.query("ticker == @ticker")
        if not tr.empty:
            fig.add_trace(go.Scattergl(
                x=tr.created_time.dt.tz_convert(ET), y=tr.yes_price, mode="markers", name="Fills",
                marker=dict(size=np.clip(np.sqrt(tr["count"]) / 3, 2, 14), color=np.where(tr.taker_side == "yes", BLUE, ORANGE),
                            opacity=0.35, line=dict(width=0)),
                customdata=np.c_[tr["count"], tr.taker_side],
                hovertemplate="%{y:.3f} × %{customdata[0]:,.0f} (taker %{customdata[1]})<extra></extra>",
            ), 1, 1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=1)

    # Row 2: underlying vs. strike and settlement.
    spot = data.get("btc_spot")
    if spot is not None:
        pad = pd.Timedelta(minutes=spot_padding_min)
        s = spot[(spot.time >= m.open_time - pad) & (spot.time <= m.close_time + pad)]
        fig.add_trace(go.Scatter(x=s.time.dt.tz_convert(ET), y=s.close, name="BTC spot",
                                 line=dict(color=INK_2, width=2)), 2, 1)
    if pd.notna(m.strike):
        fig.add_hline(y=m.strike, line=dict(color=ORANGE, width=2), row=2, col=1,
                      annotation_text=f"Strike ${m.strike:,.2f}", annotation_position="top right",
                      annotation_font_color=INK_2)
    if pd.notna(m.expiration_value):
        fig.add_trace(go.Scatter(x=[close_et], y=[m.expiration_value], mode="markers", name="Settlement avg",
                                 marker=dict(size=10, color=ORANGE, symbol="diamond",
                                             line=dict(color=SURFACE, width=2))), 2, 1)
    fig.update_yaxes(tickformat="$,.0f", row=2, col=1)

    # Row 3: volume.
    fig.add_trace(go.Bar(x=t, y=c.volume, name="Volume", marker_color=BLUE, showlegend=False,
                         marker_line_width=0), 3, 1)

    for row in (1, 2, 3):
        fig.add_vrect(x0=open_et, x1=close_et, fillcolor=GRID, opacity=0.35, line_width=0, layer="below", row=row, col=1)

    result = (m.result or "open").upper()
    move = f", moved {m.move:+,.2f}" if pd.notna(m.move) else ""
    _style(fig)
    fig.update_layout(
        title=dict(text=f"{ticker} · {open_et:%b %d %I:%M}–{close_et:%I:%M %p %Z} · result {result}{move}"
                        f" · volume {m.volume:,.0f}", y=0.98, yanchor="top"),
        height=880, bargap=0.15, margin=dict(t=120), legend=dict(y=1.06),
    )
    return fig


# ── Series overview ───────────────────────────────────────────────────────────

def implied_prob_at(markets, candles, minute):
    """YES mid-price at ``minute`` into each window (falls back to last trade)."""
    c = candles.merge(markets[["ticker", "open_time"]], on="ticker")
    c["minute"] = (c.end_period_ts - c.open_time.astype("int64") // 10**9) // 60
    c = c[c.minute == minute]
    mid = (c.yes_bid_close + c.yes_ask_close) / 2
    # A one-sided book reports bid 0 / ask 1; that "mid" is meaningless.
    spread = c.yes_ask_close - c.yes_bid_close
    prob = mid.where(spread <= 0.1, c.price_close)
    return pd.Series(prob.values, index=c.ticker.values)


def plot_calibration(markets, candles, minutes=(1, 5, 10, 14), bins=20):
    settled = markets.dropna(subset=["yes"]).set_index("ticker")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                             line=dict(color=MUTED, width=1), hoverinfo="skip"))
    for minute, color in zip(minutes, (BLUE, ORANGE, AQUA, YELLOW)):
        p = implied_prob_at(markets, candles, minute).dropna()
        df = pd.DataFrame({"p": p, "yes": settled.yes.reindex(p.index)}).dropna()
        df["bin"] = pd.cut(df.p, np.linspace(0, 1, bins + 1), include_lowest=True)
        g = df.groupby("bin", observed=True).agg(p=("p", "mean"), rate=("yes", "mean"), n=("yes", "size"))
        g = g[g.n >= 20]
        fig.add_trace(go.Scatter(
            x=g.p, y=g.rate, name=f"Minute {minute}", mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
            customdata=g.n, hovertemplate="implied %{x:.1%} → realized %{y:.1%} (n=%{customdata:,})",
        ))
    fig.update_layout(title="Calibration: market-implied P(YES) at minute k vs. realized YES rate",
                      xaxis_title="Implied probability (YES mid)", yaxis_title="Realized YES rate",
                      hovermode="closest", height=560)
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return _style(fig)


def plot_daily_volume(markets):
    s = markets.dropna(subset=["yes"])
    d = s.set_index(s.close_time.dt.tz_convert(ET)).volume.resample("D").sum()
    d = d.iloc[:-1]  # the current day is still in progress
    fig = go.Figure(go.Scatter(x=d.index, y=d.values, name="Daily volume", line=dict(color=BLUE, width=2),
                               hovertemplate="%{y:,.0f} contracts"))
    fig.update_layout(title="Contracts traded per day", yaxis_title="Contracts", height=380, showlegend=False)
    return _style(fig)


def plot_yes_rate(markets, window=7):
    s = markets.dropna(subset=["yes"])
    s = s.set_index(s.close_time.dt.tz_convert(ET))
    daily = s.yes.resample("D").agg(["mean", "size"])
    daily = daily[daily["size"] > 0]
    roll = s.yes.rolling(f"{window}D").mean().resample("D").last()
    fig = go.Figure()
    fig.add_hline(y=0.5, line=dict(color=MUTED, width=1))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["mean"], name="Daily", mode="markers",
                             marker=dict(size=8, color=BLUE, opacity=0.35, line=dict(color=SURFACE, width=2)),
                             customdata=daily["size"], hovertemplate="%{y:.1%} of %{customdata} windows"))
    fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name=f"{window}-day rolling",
                             line=dict(color=BLUE, width=2), hovertemplate="%{y:.1%}"))
    fig.update_layout(title="Share of windows resolving YES (BTC up)", height=380)
    fig.update_yaxes(tickformat=".0%")
    return _style(fig)


def plot_hour_of_day(markets):
    s = markets.dropna(subset=["yes"]).copy()
    s["hour"] = s.close_time.dt.tz_convert(ET).dt.hour
    g = s.groupby("hour").agg(yes=("yes", "mean"), volume=("volume", "median"),
                              abs_move=("move", lambda x: x.abs().median()))
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=("YES rate", "Median volume / window", "Median |BTC move| / window ($)"))
    for col, (key, fmt) in enumerate((("yes", ".1%"), ("volume", ",.0f"), ("abs_move", "$,.2f")), 1):
        fig.add_trace(go.Bar(x=g.index, y=g[key], marker_color=BLUE, marker_line_width=0, showlegend=False,
                             hovertemplate=f"%{{x}}:00 ET — %{{y:{fmt}}}<extra></extra>"), 1, col)
        fig.update_xaxes(title_text="Hour (ET)", dtick=3, row=1, col=col)
    fig.add_hline(y=0.5, line=dict(color=MUTED, width=1), row=1, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0.4, 0.6], row=1, col=1)
    fig.update_layout(title="By hour of day", height=380, bargap=0.1, hovermode="closest")
    return _style(fig)


def plot_series_overview(data=None):
    data = data or load()
    m, c = data["markets"], data["candles"]
    return [plot_calibration(m, c), plot_daily_volume(m), plot_yes_rate(m), plot_hour_of_day(m)]


def save_html(figs, path):
    figs = figs if isinstance(figs, list) else [figs]
    parts = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False) for i, f in enumerate(figs)]
    Path(path).write_text(f"<html><body style='background:{SURFACE};max-width:1200px;margin:auto'>"
                          + "".join(parts) + "</body></html>")
    print(f"wrote {path}")


if __name__ == "__main__":
    data = load()
    if len(sys.argv) > 1:
        for ticker in sys.argv[1:]:
            save_html(plot_market(ticker, data), f"{ticker}.html")
    else:
        save_html(plot_series_overview(data), "kxbtc15m_overview.html")
