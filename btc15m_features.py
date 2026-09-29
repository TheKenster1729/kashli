"""
Feature table for modeling KXBTC15M outcomes, built from ``btc15m_data.py`` output.

One row per (market, decision minute k), k = 1..14. Row k describes the world at
t_k = open_time + k minutes using only information available at t_k:

    Kalshi candle ending at t_k           covers (t_k - 60s, t_k]
    Coinbase bar starting at t_k - 60s    covers [t_k - 60s, t_k), close known at t_k

Target: ``y`` = 1 if the market resolved YES.

The benchmark every model must beat is the market itself (``mid``): the
overview plots show it is already well calibrated, so "signal" means improving
on the market's own probability, not predicting ``y`` from scratch.

Feature groups
    market   mid, spread, logit(mid), mid changes, last trade / VWAP vs mid,
             Kalshi volume (this minute, cumulative), open interest
    spot     BTC return since the window opened (Coinbase-only, so the ~$5-15
             Coinbase/BRTI basis cancels), momentum, realized vol, a random-walk
             fair probability Φ(z), Coinbase volume surge, basis at open
    context  minute k, previous window's result and move, hour of day, weekday

Execution columns (not features): yes_bid/yes_ask at t_k and at t_{k+1}, used to
simulate trading at the quote now or one minute late.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from btc15m_data import load

ET = "America/New_York"
DECISION_MINUTES = range(1, 15)

MARKET_FEATURES = ["mid", "mid_logit", "spread", "d_mid_1", "d_mid_3", "trade_vs_mid", "vwap_vs_mid",
                   "trade_range", "log_vol_min", "log_vol_cum", "log_oi"]
SPOT_FEATURES = ["ret_open", "mom_1", "mom_5", "mom_15", "vol_15", "vol_60", "z", "fair_prob",
                 "fair_logit", "cb_vol_surge", "basis_open"]
CONTEXT_FEATURES = ["k", "prev_yes", "prev_move", "hour_sin", "hour_cos", "weekday"]
FEATURES = MARKET_FEATURES + SPOT_FEATURES + CONTEXT_FEATURES


def logit(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _spot_features(spot):
    """Per-minute Coinbase features indexed by bar start time (all in basis points)."""
    s = spot.set_index("time").asfreq("1min")  # expose gaps as NaN rather than skipping them
    lr = np.log(s.close).diff() * 1e4
    out = pd.DataFrame(index=s.index)
    out["close"] = s.close
    out["bar_mean"] = (s.open + s.high + s.low + s.close) / 4
    out["mom_1"] = lr
    out["mom_5"] = lr.rolling(5).sum()
    out["mom_15"] = lr.rolling(15).sum()
    out["vol_15"] = lr.rolling(15, min_periods=10).std()
    out["vol_60"] = lr.rolling(60, min_periods=40).std()
    out["cb_vol_surge"] = np.log1p(s.volume.rolling(5).mean()) - np.log1p(s.volume.rolling(60).mean())
    return out


def build_features(data=None, max_spread=0.2, max_basis=200.0):
    data = data or load()
    markets = data["markets"].dropna(subset=["yes", "strike"]).sort_values("open_time")
    spot = _spot_features(data["btc_spot"])

    # Coinbase reference at open: mean of the minute before open, mirroring the
    # 60-second BRTI average that defines the strike.
    ref_open = spot.bar_mean.reindex(markets.open_time - pd.Timedelta("1min")).values
    markets = markets.assign(cb_ref_open=ref_open, basis_open=ref_open - markets.strike.values)
    # Drop windows whose strike disagrees wildly with spot (bad strike parses).
    markets = markets[markets.basis_open.abs() < max_basis]

    # Previous window (ends exactly at this one's open) — known at decision time.
    prev = markets.set_index("close_time")[["yes", "move", "strike"]]
    p = prev.reindex(markets.open_time)
    markets = markets.assign(prev_yes=p.yes.values, prev_move=(p.move / p.strike * 1e4).values)

    # Kalshi candles -> minute index k within each window.
    c = data["candles"].merge(markets[["ticker", "open_time"]], on="ticker")
    c["k"] = (c.end_period_ts - c.open_time.astype("int64") // 10**9) // 60
    c = c[c.k.between(1, 15)]
    c["mid"] = (c.yes_bid_close + c.yes_ask_close) / 2
    c = c.sort_values(["ticker", "k"]).reset_index(drop=True)
    g = c.groupby("ticker")
    c["d_mid_1"] = g.mid.diff(1)
    c["d_mid_3"] = g.mid.diff(3)
    c["vol_cum"] = g.volume.cumsum()
    c["next_bid"] = g.yes_bid_close.shift(-1)
    c["next_ask"] = g.yes_ask_close.shift(-1)
    c = c[c.k.isin(DECISION_MINUTES)]

    df = c.merge(markets[["ticker", "close_time", "strike", "yes", "cb_ref_open", "basis_open",
                          "prev_yes", "prev_move", "volume"]].rename(columns={"volume": "mkt_volume"}),
                 on="ticker")
    df["t"] = df.open_time + pd.to_timedelta(df.k, unit="min")

    # ── market features ──
    df["spread"] = df.yes_ask_close - df.yes_bid_close
    df["mid_logit"] = logit(df.mid)
    df["trade_vs_mid"] = (df.price_close - df.mid).fillna(0)
    df["vwap_vs_mid"] = (df.price_mean - df.mid).fillna(0)
    df["trade_range"] = (df.price_high - df.price_low).fillna(0)
    df[["d_mid_1", "d_mid_3"]] = df[["d_mid_1", "d_mid_3"]].fillna(0)
    df["log_vol_min"] = np.log1p(df.volume)
    df["log_vol_cum"] = np.log1p(df.vol_cum)
    df["log_oi"] = np.log1p(df.open_interest)

    # ── spot features (bar starting at t_k - 1min closes at t_k) ──
    sp = spot.reindex(df.t - pd.Timedelta("1min"))
    for col in ("mom_1", "mom_5", "mom_15", "vol_15", "vol_60", "cb_vol_surge"):
        df[col] = sp[col].values
    df["ret_open"] = np.log(sp.close.values / df.cb_ref_open) * 1e4
    # Random-walk fair value: settlement averages the final minute, which trims
    # about half a minute of variance off the remaining horizon.
    horizon = np.maximum(15 - df.k - 0.5, 0.5)
    df["z"] = (df.ret_open / (df.vol_60 * np.sqrt(horizon))).clip(-8, 8)
    df["fair_prob"] = norm.cdf(df.z)
    df["fair_logit"] = logit(df.fair_prob)

    # ── context ──
    hour = df.open_time.dt.tz_convert(ET).dt.hour + df.open_time.dt.tz_convert(ET).dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["weekday"] = df.open_time.dt.tz_convert(ET).dt.weekday
    df["prev_yes"] = df.prev_yes.fillna(0.5)
    df["prev_move"] = df.prev_move.fillna(0)
    df["month"] = df.open_time.dt.strftime("%Y-%m")

    # Only rows with a real two-sided book and complete spot history.
    ok = (df.yes_bid_close > 0) & (df.yes_ask_close < 1) & (df.spread <= max_spread)
    ok &= df[SPOT_FEATURES].notna().all(axis=1)
    df = df[ok].rename(columns={"yes_bid_close": "bid", "yes_ask_close": "ask"})

    cols = ["ticker", "open_time", "t", "month", "y", "bid", "ask", "next_bid", "next_ask", "mkt_volume"] + FEATURES
    return df.rename(columns={"yes": "y"})[cols].sort_values(["open_time", "k"]).reset_index(drop=True)


if __name__ == "__main__":
    f = build_features()
    f.to_parquet("data/kxbtc15m/features.parquet", index=False)
    print(f"{len(f):,} rows, {f.ticker.nunique():,} markets, {len(FEATURES)} features -> data/kxbtc15m/features.parquet")
