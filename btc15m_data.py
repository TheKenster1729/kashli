"""
Historical data collector for the Kalshi KXBTC15M ("Bitcoin price up down") series.

Every event in the series is a single 15-minute window with one binary market:
YES if the 60-second BRTI average before close is >= the 60-second average before
open (the "price to beat", stored as ``floor_strike``). The series began on
2025-12-10; this script pulls everything from then until now.

Kalshi splits data into two tiers at a moving cutoff (``GET /historical/cutoff``):
markets settled before it live under ``/historical/*``, newer ones under the
regular endpoints. Both tiers are fetched and merged here.

Datasets written to ``--out`` (default ``data/kxbtc15m``), all Parquet:

    markets.parquet   one row per 15-min window: times, strike, settlement value,
                      result, volume, open interest, final book, tier
    candles.parquet   1-minute candles per market: trade OHLC/mean, yes bid & ask
                      OHLC, volume, open interest
    btc_spot.parquet  Coinbase BTC-USD 1-minute OHLCV over the same span (a public
                      proxy for the CF Benchmarks BRTI that settles the market)
    trades/*.parquet  (opt-in, one file per UTC day) every individual fill. Recent
                      markets have ~40-55k fills each, so the full history is
                      ~1M requests; pick a range with --trades-since/--trades-until.

Re-running is incremental: finalized markets that already have candles are
skipped, spot data is appended, and trade days already on disk are skipped.

Usage:
    python btc15m_data.py                               # markets + candles + spot
    python btc15m_data.py --trades-since 2026-09-20     # also trades for that range
    python btc15m_data.py --skip-candles --skip-spot    # refresh market list only
"""

import argparse
import base64
import json
import os
import re
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

SERIES = "KXBTC15M"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
SERIES_START = datetime(2025, 12, 10, 21, 45, tzinfo=timezone.utc)  # first market's open

# Market statuses after which nothing about the market changes.
FINAL_STATUSES = {"finalized", "settled"}

# Batch candle endpoint caps: 100 tickers and 10,000 candles per request, where
# candles = tickers x window minutes (25 consecutive 15-min markets -> 25 x 375).
BATCH_MAX_MARKETS = 100
BATCH_MAX_CANDLES = 10_000


# ── HTTP ──────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Thread-safe limiter spacing calls ``interval`` seconds apart. The server
    publishes no rate-limit headers, so the interval adapts: it grows on every
    429 and shrinks back toward ``1/rate`` on successes.
    """

    def __init__(self, rate):
        self.min_interval = self.interval = 1.0 / rate
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            delay = self.next_at - now
            self.next_at = max(now, self.next_at) + self.interval
        if delay > 0:
            time.sleep(delay)

    def throttled(self):
        with self.lock:
            self.interval = min(self.interval * 1.1, 2.0)

    def ok(self):
        with self.lock:
            self.interval = max(self.min_interval, self.interval * 0.995)


class KalshiSigner:
    """
    Signs requests with an API key (RSA-PSS over timestamp + method + path).
    Only used for GETs of public market data, where it lifts the read limit
    from ~5 req/s anonymous to ~20 req/s (basic tier).
    """

    def __init__(self, key_id, key_file):
        from cryptography.hazmat.primitives import serialization
        self.key_id = key_id
        self.key = serialization.load_pem_private_key(Path(key_file).read_bytes(), password=None)

    @classmethod
    def from_env(cls, env_file=".env", prefix="PROD"):
        from dotenv import load_dotenv
        load_dotenv(env_file)
        key_id, key_file = os.getenv(f"{prefix}_KEYID"), os.getenv(f"{prefix}_KEYFILE")
        if key_id and key_file and Path(key_file).exists():
            return cls(key_id, key_file)
        return None

    def headers(self, url):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        ts = str(int(time.time() * 1000))
        sig = self.key.sign((ts + "GET" + urlparse(url).path).encode(),
                            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                            hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self.key_id, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


class Http:
    """Rate-limited GET with retry/backoff on 429 and 5xx; one Session per thread."""

    def __init__(self, rate, signer=None):
        self.limiter = RateLimiter(rate)
        self.signer = signer
        self.local = threading.local()

    @property
    def session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
            self.local.s.headers.update({"accept": "application/json"})
        return self.local.s

    def get(self, url, params=None, retries=8):
        for attempt in range(retries):
            self.limiter.wait()
            try:
                headers = self.signer.headers(url) if self.signer else None
                r = self.session.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                self.limiter.ok()
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                self.limiter.throttled()
                time.sleep(1)
                continue
            if r.status_code >= 500:
                time.sleep(min(60, 2 ** attempt))
                continue
            r.raise_for_status()
        raise RuntimeError(f"GET {url} failed after {retries} attempts")

    def paginate(self, path, key, **params):
        """Yield every item from a cursor-paginated Kalshi list endpoint."""
        cursor = None
        while True:
            q = dict(params, limit=1000)
            if cursor:
                q["cursor"] = cursor
            data = self.get(KALSHI + path, q) or {}
            items = data.get(key) or []
            yield from items
            cursor = data.get("cursor")
            if not cursor or not items:
                return


# ── Helpers ───────────────────────────────────────────────────────────────────

def num(x):
    """Kalshi returns most numbers as strings ('0.5200', '77,362.10'); '' means missing."""
    if x is None or x == "":
        return None
    return float(x.replace(",", "")) if isinstance(x, str) else float(x)


def strip_suffix(d):
    """Normalize field names across tiers: 'close_dollars'/'volume_fp' -> 'close'/'volume'."""
    return {re.sub(r"_(dollars|fp)$", "", k): v for k, v in d.items()}


def progress(label, done, total, t0):
    rate = done / max(time.time() - t0, 1e-9)
    eta = (total - done) / rate if rate else 0
    print(f"\r[{label}] {done}/{total}  {rate:.1f}/s  eta {eta/60:.1f} min", end="", flush=True)


# ── Markets ───────────────────────────────────────────────────────────────────

def _flatten_market(m, tier):
    m = strip_suffix(m)
    strike = m.get("floor_strike")
    if strike is None:
        # Early markets carry the strike only in the subtitle: "Price to beat: $92,474.75".
        hit = re.search(r"\$([\d,]+(?:\.\d+)?)", m.get("yes_sub_title") or "")
        strike = float(hit.group(1).replace(",", "")) if hit else None
    return {
        "ticker": m["ticker"],
        "event_ticker": m["event_ticker"],
        "tier": tier,
        "status": m.get("status"),
        "result": m.get("result") or None,
        "open_time": m.get("open_time"),
        "close_time": m.get("close_time"),
        "created_time": m.get("created_time"),
        "settlement_ts": m.get("settlement_ts"),
        "strike": strike,
        "expiration_value": num(m.get("expiration_value")),
        "settlement_value": num(m.get("settlement_value")),
        "volume": num(m.get("volume")),
        "open_interest": num(m.get("open_interest")),
        "liquidity": num(m.get("liquidity")),
        "last_price": num(m.get("last_price")),
        "previous_price": num(m.get("previous_price")),
        "yes_bid": num(m.get("yes_bid")),
        "yes_ask": num(m.get("yes_ask")),
        "no_bid": num(m.get("no_bid")),
        "no_ask": num(m.get("no_ask")),
        "yes_bid_size": num(m.get("yes_bid_size")),
        "yes_ask_size": num(m.get("yes_ask_size")),
        "strike_type": m.get("strike_type"),
        "price_level_structure": m.get("price_level_structure"),
        "exchange_index": m.get("exchange_index"),
        "raw": json.dumps(m, separators=(",", ":")),
    }


def fetch_markets(http):
    """Every market in the series from both tiers, deduplicated (live tier wins)."""
    rows = {}
    t0 = time.time()
    for tier, path in (("historical", "/historical/markets"), ("live", "/markets")):
        for m in http.paginate(path, "markets", series_ticker=SERIES):
            rows[m["ticker"]] = _flatten_market(m, tier)
        print(f"[markets] {tier}: {len(rows)} cumulative ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows.values())
    for c in ("open_time", "close_time", "created_time", "settlement_ts"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    df = df.sort_values("open_time").reset_index(drop=True)

    # Each window's strike is the previous window's settlement average, so fill
    # missing expiration values (common on early markets) from the next strike.
    nxt = df.set_index("open_time")["strike"]
    implied = df["close_time"].map(nxt)
    df["expiration_value_inferred"] = df["expiration_value"].isna() & implied.notna()
    df["expiration_value"] = df["expiration_value"].fillna(implied)

    df["yes"] = df["result"].map({"yes": 1.0, "no": 0.0})
    df["move"] = df["expiration_value"] - df["strike"]
    return df


# ── Candlesticks ──────────────────────────────────────────────────────────────

def _flatten_candles(ticker, candles):
    rows = []
    for c in candles:
        c = strip_suffix(c)
        row = {
            "ticker": ticker,
            "end_period_ts": c["end_period_ts"],
            "volume": num(c.get("volume")),
            "open_interest": num(c.get("open_interest")),
        }
        for block in ("price", "yes_bid", "yes_ask"):
            for k, v in strip_suffix(c.get(block) or {}).items():
                row[f"{block}_{k}"] = num(v)
        rows.append(row)
    return rows


def _historical_candles(http, m):
    data = http.get(
        f"{KALSHI}/historical/markets/{m.ticker}/candlesticks",
        {"start_ts": int(m.open_time.timestamp()), "end_ts": int(m.close_time.timestamp()),
         "period_interval": 1},
    ) or {}
    return {m.ticker: _flatten_candles(m.ticker, data.get("candlesticks") or [])}


def _live_candles(http, group):
    data = http.get(
        f"{KALSHI}/markets/candlesticks",
        {"market_tickers": ",".join(group.ticker),
         "start_ts": int(group.open_time.min().timestamp()),
         "end_ts": int(group.close_time.max().timestamp()),
         "period_interval": 1},
    ) or {}
    out = {t: [] for t in group.ticker}
    for entry in data.get("markets") or []:
        out[entry["market_ticker"]] = _flatten_candles(entry["market_ticker"], entry.get("candlesticks") or [])
    return out


def _batches(live):
    """Split open_time-sorted markets into groups that fit the batch endpoint's caps."""
    group, start = [], None
    for i, m in enumerate(live.itertuples()):
        start = m.open_time if not group else start
        minutes = (m.close_time - start).total_seconds() / 60
        if group and (len(group) + 1 > BATCH_MAX_MARKETS or (len(group) + 1) * minutes > BATCH_MAX_CANDLES):
            yield live.iloc[group]
            group, start = [], m.open_time
        group.append(i)
    if group:
        yield live.iloc[group]


def fetch_candles(http, markets, out_dir, workers):
    path = out_dir / "candles.parquet"
    done_path = out_dir / "candles_done.json"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()

    now = pd.Timestamp.now(tz="UTC")
    todo = markets[~markets.ticker.isin(done) & (markets.open_time <= now)]
    if todo.empty:
        print("[candles] up to date")
        return
    hist = todo[todo.tier == "historical"]
    live = todo[todo.tier == "live"].sort_values("open_time")
    jobs = [(_historical_candles, m) for m in hist.itertuples()]
    jobs += [(_live_candles, group) for group in _batches(live)]
    print(f"[candles] {len(todo)} markets to fetch in {len(jobs)} requests")

    results, t0 = {}, time.time()
    pool = ThreadPoolExecutor(workers)
    try:
        futures = [pool.submit(fn, http, arg) for fn, arg in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            results.update(f.result())
            if i % 50 == 0 or i == len(jobs):
                progress("candles", i, len(jobs), t0)
    finally:
        # Save whatever finished, so an interrupted run resumes where it stopped.
        pool.shutdown(wait=False, cancel_futures=True)
        print()
        new_rows = [r for rows in results.values() for r in rows]
        if new_rows:
            new = pd.DataFrame(new_rows)
            if not existing.empty:
                existing = existing[~existing.ticker.isin(results)]
            combined = pd.concat([existing, new], ignore_index=True)
            combined = combined.sort_values(["ticker", "end_period_ts"]).reset_index(drop=True)
            combined.to_parquet(path, index=False)
            print(f"[candles] {len(combined):,} rows -> {path}")
        final = set(markets.loc[markets.status.isin(FINAL_STATUSES), "ticker"])
        done |= set(results) & final
        done_path.write_text(json.dumps(sorted(done)))


# ── Trades (opt-in) ───────────────────────────────────────────────────────────

def _market_trades(http, m, cutoff):
    path = "/historical/trades" if m.close_time < cutoff else "/markets/trades"
    rows = []
    for t in http.paginate(path, "trades", ticker=m.ticker):
        t = strip_suffix(t)
        rows.append({
            "ticker": t["ticker"],
            "trade_id": t["trade_id"],
            "created_time": t["created_time"],
            "yes_price": num(t.get("yes_price")),
            "no_price": num(t.get("no_price")),
            "count": num(t.get("count")),
            "taker_side": t.get("taker_side"),
            "taker_book_side": t.get("taker_book_side"),
            "taker_outcome_side": t.get("taker_outcome_side"),
            "is_block_trade": t.get("is_block_trade"),
        })
    return rows


def fetch_trades(http, markets, out_dir, since, until, workers):
    trade_dir = out_dir / "trades"
    trade_dir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp((http.get(f"{KALSHI}/historical/cutoff") or {}).get("trades_created_ts", "1970-01-01T00:00:00Z"))
    today = pd.Timestamp.now(tz="UTC").normalize()

    sel = markets[(markets.close_time >= since) & (markets.open_time < until)
                  & markets.status.isin(FINAL_STATUSES)]
    for day, group in sel.groupby(sel.close_time.dt.normalize()):
        fp = trade_dir / f"{day:%Y-%m-%d}.parquet"
        # Today's file is rewritten each run since more markets settle during the day.
        if fp.exists() and day < today:
            continue
        rows, t0 = [], time.time()
        with ThreadPoolExecutor(workers) as pool:
            for i, r in enumerate(pool.map(lambda m: _market_trades(http, m, cutoff), group.itertuples()), 1):
                rows.extend(r)
                progress(f"trades {day:%Y-%m-%d}", i, len(group), t0)
        print(f"\n[trades] {day:%Y-%m-%d}: {len(rows):,} trades")
        if rows:
            df = pd.DataFrame(rows)
            df["created_time"] = pd.to_datetime(df.created_time, utc=True, format="ISO8601")
            df.sort_values("created_time").to_parquet(fp, index=False)


# ── BTC spot (Coinbase) ───────────────────────────────────────────────────────

def fetch_btc_spot(out_dir, end):
    """Coinbase BTC-USD 1-minute candles from SERIES_START (minus an hour) to ``end``."""
    path = out_dir / "btc_spot.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    start = (existing.time.max() + pd.Timedelta(minutes=1)) if not existing.empty \
        else pd.Timestamp(SERIES_START - timedelta(hours=1))
    if start >= end:
        print("[spot] up to date")
        return

    http = Http(rate=6)  # Coinbase public limit is ~10 req/s
    windows = pd.date_range(start, end, freq="300min")  # 300 candles max per request
    rows, t0 = [], time.time()

    def one(ws):
        we = min(ws + pd.Timedelta(minutes=299), end)
        return http.get(COINBASE, {"granularity": 60, "start": ws.isoformat(), "end": we.isoformat()}) or []

    try:
        with ThreadPoolExecutor(4) as pool:
            for i, batch in enumerate(pool.map(one, windows), 1):
                rows.extend(batch)
                if i % 20 == 0 or i == len(windows):
                    progress("spot", i, len(windows), t0)
    finally:
        print()
        if rows:
            new = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
            new["time"] = pd.to_datetime(new.pop("ts"), unit="s", utc=True)
            combined = pd.concat([existing, new[["time", "open", "high", "low", "close", "volume"]]])
            # pool.map yields in order, so an interrupted run still saved a
            # contiguous prefix and resuming from the max timestamp is safe.
            combined = combined.drop_duplicates("time").sort_values("time").reset_index(drop=True)
            combined.to_parquet(path, index=False)
            print(f"[spot] {len(combined):,} rows -> {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def load(out_dir="data/kxbtc15m"):
    """Load collected datasets as a dict of DataFrames (trades concatenated if present)."""
    out = Path(out_dir)
    data = {name: pd.read_parquet(out / f"{name}.parquet")
            for name in ("markets", "candles", "btc_spot") if (out / f"{name}.parquet").exists()}
    trade_files = sorted((out / "trades").glob("*.parquet")) if (out / "trades").exists() else []
    if trade_files:
        data["trades"] = pd.concat(map(pd.read_parquet, trade_files), ignore_index=True)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/kxbtc15m")
    p.add_argument("--rate", type=float,
                   help="Kalshi requests/sec (default 18 with an API key, 4.5 anonymous)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--env-file", default=".env", help="file with PROD_KEYID / PROD_KEYFILE")
    p.add_argument("--anonymous", action="store_true", help="don't sign requests even if keys exist")
    p.add_argument("--skip-candles", action="store_true")
    p.add_argument("--skip-spot", action="store_true")
    p.add_argument("--trades-since", help="UTC date, e.g. 2026-09-20; enables trade download")
    p.add_argument("--trades-until", help="UTC date (exclusive); default now")
    args = p.parse_args()

    # Treat `kill` like Ctrl-C so partial candle/spot progress is saved.
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    signer = None if args.anonymous else KalshiSigner.from_env(args.env_file)
    rate = args.rate or (18 if signer else 4.5)
    print(f"[http] {'authenticated' if signer else 'anonymous'}, {rate} req/s")
    http = Http(rate, signer)

    markets = fetch_markets(http)
    markets.to_parquet(out / "markets.parquet", index=False)
    settled = markets.result.notna().sum()
    print(f"[markets] {len(markets):,} markets ({settled:,} settled), "
          f"{markets.open_time.min():%Y-%m-%d %H:%M} -> {markets.close_time.max():%Y-%m-%d %H:%M} UTC")

    if not args.skip_candles:
        fetch_candles(http, markets, out, args.workers)
    if not args.skip_spot:
        fetch_btc_spot(out, pd.Timestamp.now(tz="UTC").floor("min"))
    if args.trades_since:
        since = pd.Timestamp(args.trades_since, tz="UTC")
        until = pd.Timestamp(args.trades_until, tz="UTC") if args.trades_until else pd.Timestamp.now(tz="UTC")
        fetch_trades(http, markets, out, since, until, args.workers)


if __name__ == "__main__":
    main()
