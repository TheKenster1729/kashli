"""
Baseline models for KXBTC15M: does anything beat the market's own price?

Walk-forward by month: for each test month, fit on every earlier month, predict
the test month. No shuffling, so nothing from the future leaks into training.

Models
    market      the YES mid-price itself (the benchmark)
    market_lr   logistic recalibration of the mid alone (is the market merely
                miscalibrated, or is there information it ignores?)
    fair        random-walk probability Φ(z) from Coinbase spot alone, no fitting
    lr          logistic regression on all features (standardized)
    rf          random forest on all features

Evaluation
    Log loss / Brier / AUC per month and per decision minute, with the paired
    difference vs. the market bootstrapped over markets (rows within one window
    are correlated, so markets are the independent unit).

    Trading simulation: in each window, take the first minute where the model's
    edge after Kalshi's taker fee (0.07 · P · (1 − P) per contract) exceeds a
    threshold, buy one contract at the ask (YES) or at 1 − bid (NO), hold to
    settlement. Run twice: filling at the quote at t_k, and at the quote one
    minute later, since an edge that vanishes with a minute's delay is a
    latency race rather than a forecast.

Usage:
    python btc15m_models.py                    # -> kxbtc15m_models.html + stdout report
    python btc15m_models.py --first-test 2026-06
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from btc15m_features import FEATURES, build_features
from btc15m_plots import AQUA, BLUE, INK_2, MUTED, ORANGE, _style, save_html

FEE_RATE = 0.07
EPS = 1e-3

# numpy 2.x on macOS Accelerate raises spurious FP warnings in matmul (they appear
# on random data too, and fitted coefficients are finite and correct).
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
FITTED = {"market_lr": ["mid_logit"], "lr": FEATURES, "rf": FEATURES}
COLORS = {"market_lr": AQUA, "lr": BLUE, "rf": ORANGE}


def make_model(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=200, max_features=0.33,
                                      max_samples=0.3, n_jobs=-1, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(df, first_test):
    """Out-of-sample predictions for every row in months >= first_test."""
    months = sorted(m for m in df.month.unique() if m >= first_test)
    preds, fitted = [], {}
    for month in months:
        train, test = df[df.month < month], df[df.month == month]
        out = test[["ticker", "open_time", "month", "k", "y", "bid", "ask", "next_bid", "next_ask", "mid"]].copy()
        out["p_market"] = test.mid.clip(EPS, 1 - EPS)
        out["p_fair"] = test.fair_prob.clip(EPS, 1 - EPS)
        t0 = time.time()
        for name, cols in FITTED.items():
            model = make_model(name).fit(train[cols], train.y)
            out[f"p_{name}"] = model.predict_proba(test[cols])[:, 1].clip(EPS, 1 - EPS)
            fitted[name] = model
        print(f"[fit] {month}: train {len(train):,} rows, test {len(test):,} rows ({time.time()-t0:.0f}s)")
        preds.append(out)
    # Models from the last fold are returned for inspecting coefficients/importances.
    return pd.concat(preds, ignore_index=True), fitted


def model_names(preds):
    return [c[2:] for c in preds.columns if c.startswith("p_")]


# ── Scoring ───────────────────────────────────────────────────────────────────

def row_log_loss(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scores(preds, by=None):
    rows = []
    groups = preds.groupby(by) if by else [("all", preds)]
    for key, g in groups:
        for name in model_names(preds):
            p = g[f"p_{name}"]
            rows.append({"group": key, "model": name, "n": len(g),
                         "log_loss": log_loss(g.y, p, labels=[0, 1]), "brier": brier_score_loss(g.y, p),
                         "auc": roc_auc_score(g.y, p) if g.y.nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def bootstrap_vs_market(preds, name, n_boot=1000, seed=0):
    """Mean per-row log-loss improvement over the market, with a 95% CI resampled by market."""
    diff = row_log_loss(preds.y, preds.p_market) - row_log_loss(preds.y, preds[f"p_{name}"])
    per_mkt = diff.groupby(preds.ticker).agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_mkt), (n_boot, len(per_mkt)))
    s, n = per_mkt["sum"].values, per_mkt["size"].values
    boots = s[idx].sum(1) / n[idx].sum(1)
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ── Trading simulation ────────────────────────────────────────────────────────

def fee(price):
    return FEE_RATE * price * (1 - price)


def simulate(preds, name, threshold, delayed=False):
    """One contract per window at the first minute whose net edge exceeds ``threshold``."""
    p = preds[f"p_{name}"]
    edge_yes = p - preds.ask - fee(preds.ask)
    edge_no = (1 - p) - (1 - preds.bid) - fee(1 - preds.bid)
    side_yes = edge_yes >= edge_no
    edge = np.where(side_yes, edge_yes, edge_no)
    cand = preds.assign(side_yes=side_yes, edge=edge)[edge > threshold]
    trades = cand.sort_values(["open_time", "k"]).groupby("ticker", sort=False).head(1).copy()

    bid, ask = (trades.next_bid, trades.next_ask) if delayed else (trades.bid, trades.ask)
    price = np.where(trades.side_yes, ask, 1 - bid)
    won = np.where(trades.side_yes, trades.y, 1 - trades.y)
    trades["price"] = price
    trades["pnl"] = won - price - fee(price)
    return trades.dropna(subset=["pnl"])


def pnl_summary(trades, n_boot=1000, seed=0):
    if trades.empty:
        return {"trades": 0, "mean_pnl": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "total": 0.0, "win_rate": np.nan}
    x = trades.pnl.values
    boots = np.random.default_rng(seed).choice(x, (n_boot, len(x))).mean(1)
    return {"trades": len(x), "mean_pnl": x.mean(), "ci_lo": np.percentile(boots, 2.5),
            "ci_hi": np.percentile(boots, 97.5), "total": x.sum(), "win_rate": (x > 0).mean()}


def trading_table(preds, thresholds):
    rows = []
    for name in ("market_lr", "fair", "lr", "rf"):
        for thr in thresholds:
            for delayed in (False, True):
                rows.append({"model": name, "threshold": thr, "fill": "t+1min" if delayed else "t",
                             **pnl_summary(simulate(preds, name, thr, delayed))})
    return pd.DataFrame(rows)


# ── Importances ───────────────────────────────────────────────────────────────

def importances(fitted, df, last_month, n_sample=20_000):
    lr = fitted["lr"]
    coef = pd.Series(lr[-1].coef_[0], index=FEATURES, name="lr_coef_std")
    test = df[df.month == last_month]
    test = test.sample(min(n_sample, len(test)), random_state=0)
    perm = permutation_importance(fitted["rf"], test[FEATURES], test.y, scoring="neg_log_loss",
                                  n_repeats=3, random_state=0, n_jobs=-1)
    rf = pd.Series(perm.importances_mean, index=FEATURES, name="rf_perm_logloss")
    return pd.concat([coef, rf], axis=1).sort_values("rf_perm_logloss", ascending=False)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_delta_by(preds, by, title, xtitle):
    s = scores(preds, by)
    base = s[s.model == "market"].set_index("group").log_loss
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    for name, color in COLORS.items():
        d = s[s.model == name].set_index("group")
        gain = (base - d.log_loss) * 1000
        fig.add_trace(go.Scatter(x=[str(g) for g in gain.index], y=gain.values, name=name, mode="lines+markers",
                                 line=dict(color=color, width=2), marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
                                 hovertemplate="%{y:+.2f} mnats/row"))
    fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title="Log-loss improvement vs market (millinats)",
                      height=420)
    return _style(fig)


def plot_pnl(preds, threshold):
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    for name in ("lr", "rf"):
        for delayed in (False, True):
            t = simulate(preds, name, threshold, delayed).sort_values("open_time")
            fig.add_trace(go.Scatter(
                x=t.open_time, y=t.pnl.cumsum(), name=f"{name}, fill at {'t+1min' if delayed else 't'}",
                line=dict(color=COLORS[name], width=2, dash="dot" if delayed else "solid"),
                hovertemplate="$%{y:,.2f} after %{pointNumber} trades"))
    fig.update_layout(title=f"Cumulative P&L, 1 contract/window, net edge > {threshold:.0%} after fees",
                      yaxis_title="$ per 1-contract strategy", height=420)
    fig.update_yaxes(tickformat="$,.0f")
    return _style(fig)


def plot_importance(imp, top=15):
    d = imp.rf_perm_logloss.head(top)[::-1] * 1000
    fig = go.Figure(go.Bar(x=d.values, y=d.index, orientation="h", marker_color=BLUE, marker_line_width=0,
                           hovertemplate="%{y}: %{x:.2f} mnats<extra></extra>"))
    fig.update_layout(title="Random forest: permutation importance (log-loss increase when shuffled, last fold)",
                      xaxis_title="millinats", height=480, hovermode="closest", showlegend=False)
    fig.update_yaxes(tickfont=dict(color=INK_2))
    return _style(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-test", default="2026-03", help="first walk-forward test month (YYYY-MM)")
    ap.add_argument("--thresholds", default="0,0.02,0.05")
    ap.add_argument("--out", default="kxbtc15m_models.html")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]
    pd.set_option("display.width", 160, "display.max_columns", 20)

    df = build_features()
    print(f"[data] {len(df):,} rows, {df.ticker.nunique():,} markets, {df.month.min()} -> {df.month.max()}")
    preds, fitted = walk_forward(df, args.first_test)
    preds.to_parquet("data/kxbtc15m/oos_predictions.parquet", index=False)

    print("\n== Out-of-sample, all test months ==")
    print(scores(preds).drop(columns="group").set_index("model").round(4))
    print("\nLog-loss improvement vs market per row (95% CI, bootstrapped over markets):")
    for name in ("market_lr", "fair", "lr", "rf"):
        m, lo, hi = bootstrap_vs_market(preds, name)
        print(f"  {name:10s} {m*1000:+7.2f} mnats  [{lo*1000:+.2f}, {hi*1000:+.2f}]")

    s = scores(preds, "k")
    print("\n== Log loss by decision minute ==")
    print(s.pivot(index="group", columns="model", values="log_loss").round(4))

    print("\n== Trading simulation (1 contract per window) ==")
    print(trading_table(preds, thresholds).round(4).to_string(index=False))

    imp = importances(fitted, df, preds.month.max())
    print("\n== Feature importance (last fold) ==")
    print(imp.round(4))

    save_html([
        plot_delta_by(preds, "month", "Out-of-sample log-loss improvement over the market, by month", "Test month"),
        plot_delta_by(preds, "k", "…by minute into the window", "Decision minute k"),
        plot_pnl(preds, 0.02),
        plot_importance(imp),
    ], args.out)


if __name__ == "__main__":
    main()
