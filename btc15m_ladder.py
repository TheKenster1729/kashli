"""
Cross-market consistency: 15-minute markets vs. the hourly ladder.

The 15-minute window ending on the hour (e.g. KXBTC15M-26SEP232300-00) and the
hourly above/below ladder for that hour (KXBTCD-26SEP2323-T…) settle on the
*same number*: the 60-second BRTI average before the hour.

    15-min YES   <=>  X >= s          (s = the window's strike)
    ladder K YES <=>  X >  K

So for ladder strikes K_lo < s <= K_hi, P(X > K_hi) <= P15 <= P(X > K_lo) must
hold, and a price outside those bounds is an arbitrage:

    bid15 > ask(K_lo)  ->  buy NO on the 15-min at 1 - bid15, buy YES on K_lo
    bid(K_hi) > ask15  ->  buy YES on the 15-min at ask15, buy NO on K_hi
                           each pays >= $1 in every outcome; profit = the gap - fees

Beyond hard bounds, the ladder implies a price for the 15-min contract
(interpolating between strikes in probit space). When the two disagree, we
ask which one the outcome sides with.

Usage:
    python btc15m_ladder.py collect      # ladder strikes + 1-min candles (~50k requests)
    python btc15m_ladder.py analyze      # -> kxbtc15m_ladder.html + stdout report
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from btc15m_data import KALSHI, Http, KalshiSigner, _flatten_market, fetch_candles, load
from btc15m_models import fee
from btc15m_plots import BLUE, MUTED, ORANGE, _style, save_html

LADDER = "KXBTCD"
OUT = Path("data/kxbtcd")
STRIKES_EACH_SIDE = 4
EPS = 1e-3


# ── Collection ────────────────────────────────────────────────────────────────

def hourly_windows(markets):
    """15-min markets whose window ends on the hour, with the matching ladder event ticker."""
    m = markets[(markets.close_time.dt.minute == 0) & markets.strike.notna()].copy()
    # Both tickers encode the close in ET: KXBTC15M-26SEP232300 <-> KXBTCD-26SEP2323.
    m["ladder_event"] = LADDER + "-" + m.event_ticker.str.split("-").str[1].str[:-2]
    return m


def fetch_ladder(http, event, strike, cutoff):
    """The ladder strikes nearest ``strike`` for one hourly event."""
    order = ("historical", "live") if pd.Timestamp(event.close_time) < cutoff else ("live", "historical")
    for tier in order:
        path = "/historical/markets" if tier == "historical" else "/markets"
        data = http.get(KALSHI + path, {"event_ticker": event.ladder_event, "limit": 1000}) or {}
        ms = data.get("markets") or []
        if ms:
            break
    else:
        return []
    rows = [_flatten_market(m, tier) for m in ms]
    rows = [r for r in rows if r["strike"] is not None]
    below = sorted((r for r in rows if r["strike"] < strike), key=lambda r: -r["strike"])[:STRIKES_EACH_SIDE]
    above = sorted((r for r in rows if r["strike"] >= strike), key=lambda r: r["strike"])[:STRIKES_EACH_SIDE]
    for r in below + above:
        r["window_ticker"] = event.ticker
        r["n_strikes"] = len(rows)
    return below + above


def collect(workers=12):
    OUT.mkdir(parents=True, exist_ok=True)
    signer = KalshiSigner.from_env()
    http = Http(18 if signer else 4.5, signer)
    windows = hourly_windows(load()["markets"])
    windows = windows[windows.status.isin({"finalized", "settled"})]

    path = OUT / "ladder_markets.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["window_ticker"])
    todo = windows[~windows.ticker.isin(existing.window_ticker)]
    cutoff = pd.Timestamp((http.get(f"{KALSHI}/historical/cutoff") or {})["market_settled_ts"])
    print(f"[ladder] {len(windows):,} hourly windows, {len(todo):,} to fetch")

    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for i, r in enumerate(pool.map(lambda e: fetch_ladder(http, e, e.strike, cutoff), todo.itertuples()), 1):
            rows.extend(r)
            if i % 200 == 0:
                print(f"\r[ladder] {i}/{len(todo)} events", end="", flush=True)
    print()
    if rows:
        new = pd.DataFrame(rows)
        for c in ("open_time", "close_time", "created_time", "settlement_ts"):
            new[c] = pd.to_datetime(new[c], utc=True, format="ISO8601")
        lad = pd.concat([existing, new], ignore_index=True) if len(existing) else new
        lad.to_parquet(path, index=False)
    lad = pd.read_parquet(path)
    print(f"[ladder] {len(lad):,} ladder markets for {lad.window_ticker.nunique():,} windows -> {path}")

    # Only the final 15 minutes overlap the 15-min window; fetch just those.
    window = lad.assign(open_time=lad.close_time - pd.Timedelta(minutes=15))
    fetch_candles(http, window, OUT, workers)


# ── Analysis ──────────────────────────────────────────────────────────────────

def _minute_quotes(candles, open_time):
    c = candles.merge(open_time, on="ticker")
    c["k"] = (c.end_period_ts - c.win_open.astype("int64") // 10**9) // 60
    c = c[c.k.between(1, 14)]
    return c.rename(columns={"yes_bid_close": "bid", "yes_ask_close": "ask"})[["ticker", "k", "bid", "ask"]]


def build_panel():
    """One row per (hourly window, minute k, ladder strike) with both markets' quotes."""
    data = load()
    windows = hourly_windows(data["markets"]).dropna(subset=["yes"])
    lad = pd.read_parquet(OUT / "ladder_markets.parquet")
    lad_c = pd.read_parquet(OUT / "candles.parquet")

    w_open = windows[["ticker", "open_time"]].rename(columns={"open_time": "win_open"})
    q15 = _minute_quotes(data["candles"][data["candles"].ticker.isin(windows.ticker)], w_open)
    q15 = q15.merge(windows[["ticker", "strike", "yes", "expiration_value"]], on="ticker")
    q15 = q15.rename(columns={"ticker": "window_ticker", "bid": "bid15", "ask": "ask15"})

    l_open = lad[["ticker", "window_ticker"]].merge(w_open.rename(columns={"ticker": "window_ticker"}), on="window_ticker")
    ql = _minute_quotes(lad_c, l_open[["ticker", "win_open"]])
    ql = ql.merge(lad[["ticker", "window_ticker", "strike", "result", "expiration_value"]]
                  .rename(columns={"strike": "K", "expiration_value": "lad_expiration"}), on="ticker")

    panel = q15.merge(ql.rename(columns={"ticker": "lad_ticker"}), on=["window_ticker", "k"])
    # Drop one-sided books (bid 0 / ask 1), which quote nothing.
    ok = (panel.bid15 > 0) & (panel.ask15 < 1) & (panel.bid > 0) & (panel.ask < 1)
    return panel[ok].reset_index(drop=True), windows, lad


def arbitrage(panel):
    """Hard-bound violations, net of taker fees on both legs, per snapshot."""
    lo = panel[panel.K < panel.strike]      # X > K_lo is implied by X >= s
    hi = panel[panel.K >= panel.strike]     # X > K_hi implies X >= s
    a_lo = lo.assign(side="sell15/buy K_lo",
                     profit=lo.bid15 - lo.ask - fee(1 - lo.bid15) - fee(lo.ask))
    a_hi = hi.assign(side="buy15/sell K_hi",
                     profit=hi.bid - hi.ask15 - fee(hi.ask15) - fee(1 - hi.bid))
    a = pd.concat([a_lo, a_hi])
    # Best opportunity per (window, minute): the tightest bound is what matters.
    return a.sort_values("profit").groupby(["window_ticker", "k"]).tail(1)


def implied(panel):
    """Ladder-implied P(X >= s) per (window, minute), by probit interpolation between bracketing strikes."""
    p = panel.assign(mid=(panel.bid + panel.ask) / 2)
    lo = p[p.K < p.strike].sort_values("K").groupby(["window_ticker", "k"]).tail(1)
    hi = p[p.K >= p.strike].sort_values("K").groupby(["window_ticker", "k"]).head(1)
    cols = ["window_ticker", "k", "K", "mid"]
    b = lo[cols + ["strike", "bid15", "ask15", "yes"]].merge(hi[cols], on=["window_ticker", "k"], suffixes=("_lo", "_hi"))
    z_lo, z_hi = norm.ppf(b.mid_lo.clip(EPS, 1 - EPS)), norm.ppf(b.mid_hi.clip(EPS, 1 - EPS))
    w = (b.strike - b.K_lo) / (b.K_hi - b.K_lo)
    b["p_ladder"] = norm.cdf(z_lo + w * (z_hi - z_lo)).clip(EPS, 1 - EPS)
    b["mid15"] = ((b.bid15 + b.ask15) / 2).clip(EPS, 1 - EPS)
    b["dev"] = b.mid15 - b.p_ladder
    b["gap"] = b.K_hi - b.K_lo
    return b


def ll(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def report(panel, windows, lad):
    print(f"[panel] {panel.window_ticker.nunique():,} hourly windows, {len(panel):,} (window, minute, strike) rows")

    # Same settlement number? Compare recorded settlement values where both exist.
    s = lad.drop_duplicates("window_ticker").merge(
        windows[["ticker", "expiration_value"]].rename(columns={"ticker": "window_ticker", "expiration_value": "v15"}),
        on="window_ticker")
    s = s.dropna(subset=["expiration_value", "v15"])
    print(f"[check] settlement values identical in {(abs(s.expiration_value - s.v15) < 0.01).mean():.1%} "
          f"of {len(s):,} windows")

    a = arbitrage(panel)
    print("\n== Hard-bound arbitrage (taker on both legs, 1-min snapshot quotes) ==")
    for thr in (0, 0.01, 0.02, 0.05):
        x = a[a.profit > thr]
        print(f"  profit > ${thr:.2f}: {len(x):6,} snapshots in {x.window_ticker.nunique():5,} windows "
              f"({x.window_ticker.nunique() / a.window_ticker.nunique():.1%}), median profit ${x.profit.median() if len(x) else 0:.3f}")

    b = implied(panel)
    print(f"\n== 15-min mid vs ladder-implied price ({len(b):,} snapshots) ==")
    print(f"  |dev| median {b.dev.abs().median():.3f}, 90th pct {b.dev.abs().quantile(.9):.3f}, "
          f"median strike gap ${b.gap.median():.0f}")
    print(f"  log loss: 15-min mid {ll(b.yes, b.mid15).mean():.4f} | ladder-implied {ll(b.yes, b.p_ladder).mean():.4f} "
          f"| average {ll(b.yes, (b.mid15 + b.p_ladder) / 2).mean():.4f}")
    # Regress the 15-min market's forecast error on the disagreement:
    # slope 0 -> the 15-min price was right, slope -1 -> the ladder was right.
    slope = np.polyfit(b.dev, b.yes - b.mid15, 1)[0]
    print(f"  slope of (outcome - mid15) on dev: {slope:+.2f}  (0 = 15-min right, -1 = ladder right)")

    print("\n  Trade the 15-min toward the ladder when |dev| > threshold (taker, hold to settlement):")
    for thr in (0.02, 0.05, 0.1):
        x = b[b.dev.abs() > thr].sort_values("k").groupby("window_ticker").head(1)
        buy = x.dev < 0
        price = np.where(buy, x.ask15, 1 - x.bid15)
        won = np.where(buy, x.yes, 1 - x.yes)
        pnl = won - price - fee(price)
        ci = np.percentile(np.random.default_rng(0).choice(pnl, (1000, len(pnl))).mean(1), [2.5, 97.5]) if len(pnl) else [np.nan] * 2
        print(f"    |dev| > {thr:.2f}: {len(x):5,} trades, mean ${pnl.mean() if len(pnl) else np.nan:+.4f} "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
    return a, b


def plot_dev_vs_outcome(b, bins=16):
    b = b[b.dev.abs() < 0.3]
    b = b.assign(bin=pd.cut(b.dev, bins))
    g = b.groupby("bin", observed=True).agg(dev=("dev", "mean"), err=("yes", "mean"), mid=("mid15", "mean"), n=("dev", "size"))
    g["err"] = g.err - g.mid
    g = g[g.n >= 30]
    lim = float(b.dev.abs().max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[lim, -lim], mode="lines", name="Ladder right",
                             line=dict(color=MUTED, width=1), hoverinfo="skip"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.add_trace(go.Scatter(x=g.dev, y=g.err, mode="lines+markers", name="Observed", line=dict(color=BLUE, width=2),
                             marker=dict(size=8, line=dict(color="#fcfcfb", width=2)), customdata=g.n,
                             hovertemplate="dev %{x:+.3f} → outcome − mid15 %{y:+.3f} (n=%{customdata:,})<extra></extra>"))
    fig.update_layout(title="When the 15-min price and the ladder disagree, who is right?",
                      xaxis_title="15-min mid − ladder-implied price", yaxis_title="Outcome − 15-min mid (mean)",
                      height=480, hovermode="closest")
    return _style(fig)


def plot_arb_over_time(a, thr=0.0):
    x = a[a.profit > thr]
    d = x.groupby(pd.to_datetime(x.window_ticker.str.split("-").str[1].str[:7], format="%y%b%d")).window_ticker.nunique()
    n = a.groupby(pd.to_datetime(a.window_ticker.str.split("-").str[1].str[:7], format="%y%b%d")).window_ticker.nunique()
    share = (d.reindex(n.index).fillna(0) / n).rolling(7, min_periods=1).mean()
    fig = go.Figure(go.Scatter(x=share.index, y=share.values, line=dict(color=ORANGE, width=2),
                               hovertemplate="%{y:.1%} of hourly windows"))
    fig.update_layout(title=f"Share of hourly windows with a fee-positive bound violation (profit > ${thr:.2f}), 7-day avg",
                      height=380, showlegend=False)
    fig.update_yaxes(tickformat=".0%")
    return _style(fig)


def analyze(out="kxbtc15m_ladder.html"):
    panel, windows, lad = build_panel()
    a, b = report(panel, windows, lad)
    save_html([plot_dev_vs_outcome(b), plot_arb_over_time(a)], out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["collect", "analyze"])
    args = ap.parse_args()
    collect() if args.command == "collect" else analyze()
