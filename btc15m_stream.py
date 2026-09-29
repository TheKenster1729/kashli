"""
Live recorder for Kalshi crypto markets and their underlyings, for studying
sub-minute behavior (overreaction, lead/lag, cross-market pricing) that the
REST history can't show. Historical data at this resolution doesn't exist, so
every day this isn't running is lost.

Streams (all via Kalshi's authenticated WebSocket unless noted):

    orderbook_delta   full order book: snapshot on subscribe, then every change
    trade             every fill
    ticker            price / bid / ask / volume / open-interest updates
    cfbenchmarks_value_5hz   the CF Benchmarks real-time indices (BRTI etc.)
                      that *settle* these markets, ~5 ticks/sec
    Coinbase          BTC-USD / ETH-USD trades and best bid/ask (public feed)

Markets are discovered over REST every 30 s and added to / removed from the
live subscription, so new 15-minute windows are picked up as they open. Hourly
ladders (KXBTCD) are limited to events closing within ``--ladder-hours`` and
strikes within ``--ladder-band`` dollars of the live index, since far strikes
generate order-book churn but carry no information for our purposes.

Output: gzip JSON lines, one file per source per UTC hour:
    data/stream/kalshi/2026-09-24/20.jsonl.gz
    data/stream/coinbase/2026-09-24/20.jsonl.gz
Each line is {"t": <local receive time, unix ms>, "m": <message>}. Redundant
fields (market_id, ISO timestamps duplicating ts_ms) are dropped to save space.
The recorder stops itself if free disk space falls below ``--min-free-gb``.

Usage:
    python btc15m_stream.py                         # run until stopped
    nohup python btc15m_stream.py > data/stream/recorder.log 2>&1 &
    python btc15m_stream.py --duration 300          # 5-minute test
"""

import argparse
import asyncio
import gzip
import json
import logging
import os
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import websockets

from btc15m_data import KALSHI, KalshiSigner

KALSHI_WS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
COINBASE_REST = "https://api.exchange.coinbase.com/products"
INDEX_PRODUCTS = {"BRTI": "BTC-USD", "ETHUSD_RTI": "ETH-USD", "SOLUSD_RTI": "SOL-USD"}
MARKET_CHANNELS = ["orderbook_delta", "trade", "ticker"]
DROP_FIELDS = {"market_id", "ts"}  # "ts" is an ISO copy of "ts_ms"

log = logging.getLogger("stream")


# ── Storage ───────────────────────────────────────────────────────────────────

class HourlyWriter:
    """Appends JSON lines to data/stream/<source>/<date>/<hour>.jsonl.gz."""

    def __init__(self, root, source):
        self.dir = Path(root) / source
        self.source = source
        self.fh = None
        self.hour = None
        self.count = 0
        self.last_flush = time.monotonic()

    def write(self, msg):
        now = datetime.now(timezone.utc)
        hour = now.strftime("%Y-%m-%d/%H")
        if hour != self.hour:
            self.close()
            path = self.dir / f"{hour}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode: a restart within the same hour adds a new gzip member,
            # which gzip readers concatenate transparently.
            self.fh = gzip.open(path, "at", compresslevel=6)
            self.hour = hour
        self.fh.write(json.dumps({"t": int(now.timestamp() * 1000), "m": msg}, separators=(",", ":")) + "\n")
        self.count += 1
        if time.monotonic() - self.last_flush > 5:
            self.fh.flush()
            self.last_flush = time.monotonic()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


def read_stream(root, source, start=None, end=None):
    """Yield (receive_ms, message) from recorded files, in file order."""
    for path in sorted((Path(root) / source).glob("*/*.jsonl.gz")):
        stamp = path.parent.name + "/" + path.stem.split(".")[0]
        if (start and stamp < start) or (end and stamp > end):
            continue
        with gzip.open(path, "rt") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    break  # truncated tail from a hard kill
                yield rec["t"], rec["m"]


# ── Market discovery ──────────────────────────────────────────────────────────

class Discovery:
    """Decides which market tickers to track, via REST polling."""

    def __init__(self, series_15m, ladder_series, ladder_hours, ladder_band):
        self.series_15m = series_15m
        self.ladder_series = ladder_series
        self.ladder_hours = ladder_hours
        self.ladder_band = ladder_band
        self.session = requests.Session()
        self.index_value = {}  # e.g. {"BRTI": 84210.5}, fed by the index stream

    def _get(self, path, **params):
        r = self.session.get(KALSHI + path, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _coinbase_spot(self, index_id):
        """Fallback spot price before the index stream has delivered a value."""
        product = INDEX_PRODUCTS.get(index_id)
        if not product:
            return None
        try:
            return float(self.session.get(f"{COINBASE_REST}/{product}/ticker", timeout=10).json()["price"])
        except Exception:
            return None

    def tickers(self):
        now = datetime.now(timezone.utc)
        out = set()
        ts = int(now.timestamp())
        for series in self.series_15m:
            # The current window plus the next one (listed a day ahead), so the
            # next book is recorded from the moment it starts forming.
            for m in self._get("/markets", series_ticker=series, min_close_ts=ts, max_close_ts=ts + 31 * 60,
                               limit=50)["markets"]:
                opens = datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
                if (opens - now).total_seconds() < 20 * 60:
                    out.add(m["ticker"])
        for series, index_id in self.ladder_series.items():
            spot = self.index_value.get(index_id) or self._coinbase_spot(index_id)
            events = self._get("/events", series_ticker=series, status="open", limit=50)["events"]
            for ev in events:
                markets = self._get("/markets", event_ticker=ev["event_ticker"], limit=1000)["markets"]
                if not markets:
                    continue
                closes = datetime.fromisoformat(markets[0]["close_time"].replace("Z", "+00:00"))
                if (closes - now).total_seconds() > self.ladder_hours * 3600:
                    continue
                for m in markets:
                    strike = m.get("floor_strike") or m.get("cap_strike")
                    if spot is None or strike is None or abs(strike - spot) <= self.ladder_band:
                        out.add(m["ticker"])
        return out


# ── Kalshi ────────────────────────────────────────────────────────────────────

class KalshiRecorder:
    def __init__(self, signer, discovery, writer, index_ids):
        self.signer = signer
        self.discovery = discovery
        self.writer = writer
        self.index_ids = index_ids
        self.ws = None
        self.msg_id = 0
        self.sid_channel = {}      # sid -> channel
        self.pending = {}          # command id -> channel
        self.tracked = set()
        self.last_seq = {}         # sid -> last seq seen

    async def send(self, cmd, params, channel=None):
        self.msg_id += 1
        if channel:
            self.pending[self.msg_id] = channel
        await self.ws.send(json.dumps({"id": self.msg_id, "cmd": cmd, "params": params}))

    def market_sids(self):
        return [sid for sid, ch in self.sid_channel.items() if ch in MARKET_CHANNELS]

    async def refresh_markets(self):
        """Poll REST for the desired market set and diff it into the live subscriptions."""
        while True:
            await asyncio.sleep(30)
            try:
                want = await asyncio.to_thread(self.discovery.tickers)
            except Exception as e:
                log.warning("discovery failed: %r", e)
                continue
            add, drop = sorted(want - self.tracked), sorted(self.tracked - want)
            for action, tickers in (("add_markets", add), ("delete_markets", drop)):
                if tickers:
                    for sid in self.market_sids():
                        await self.send("update_subscription", {"sid": sid, "market_tickers": tickers, "action": action})
            if add or drop:
                log.info("markets +%d -%d (tracking %d)", len(add), len(drop), len(want))
                self.writer.write({"type": "recorder_markets", "add": add, "drop": drop})
            self.tracked = want

    def handle(self, raw):
        msg = json.loads(raw)
        typ = msg.get("type")
        body = msg.get("msg")
        if isinstance(body, dict):
            for f in DROP_FIELDS & body.keys():
                del body[f]
        if typ == "subscribed":
            self.sid_channel[body["sid"]] = body["channel"]
            self.pending.pop(msg.get("id"), None)
        elif typ == "error":
            log.warning("server error: %s", raw[:300])
        elif typ == "cfbenchmarks_value_5hz":
            self.discovery.index_value[body["index_id"]] = float(body["value_usd"])
            body.pop("data", None)  # raw upstream frame duplicating value/time

        # Sequence gaps on the order book mean our reconstructed book is wrong
        # from here on; mark it and request a fresh snapshot. Command replies
        # ("ok") consume sequence numbers too, so every message on the sid counts.
        sid, seq = msg.get("sid"), msg.get("seq")
        if sid is not None and seq is not None and self.sid_channel.get(sid) == "orderbook_delta":
            prev = self.last_seq.get(sid)
            if prev is not None and seq != prev + 1:
                log.warning("seq gap on sid %s: %s -> %s", sid, prev, seq)
                self.writer.write({"type": "recorder_gap", "sid": sid, "from": prev, "to": seq})
                asyncio.get_running_loop().create_task(
                    self.send("update_subscription", {"sid": sid, "market_tickers": sorted(self.tracked),
                                                      "action": "get_snapshot"}))
            self.last_seq[sid] = seq
        self.writer.write(msg)

    async def run_once(self):
        self.tracked = await asyncio.to_thread(self.discovery.tickers)
        self.sid_channel.clear()
        self.last_seq.clear()
        async with websockets.connect(KALSHI_WS, additional_headers=self.signer.headers(KALSHI_WS),
                                      max_size=None, ping_interval=20, ping_timeout=30) as ws:
            self.ws = ws
            log.info("kalshi connected, tracking %d markets", len(self.tracked))
            self.writer.write({"type": "recorder_connect", "markets": sorted(self.tracked)})
            await self.send("subscribe", {"channels": ["cfbenchmarks_value_5hz"], "index_ids": self.index_ids})
            if self.tracked:
                await self.send("subscribe", {"channels": MARKET_CHANNELS, "market_tickers": sorted(self.tracked)})
            refresher = asyncio.create_task(self.refresh_markets())
            try:
                async for raw in ws:
                    self.handle(raw)
            finally:
                refresher.cancel()

    async def run(self):
        backoff = 1
        while True:
            t0 = time.monotonic()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep recording through any failure; just reconnect
                log.warning("kalshi disconnected: %r", e)
            self.writer.write({"type": "recorder_disconnect"})
            backoff = 1 if time.monotonic() - t0 > 60 else min(backoff * 2, 60)
            await asyncio.sleep(backoff)


# ── Coinbase ──────────────────────────────────────────────────────────────────

async def coinbase(writer, products):
    backoff = 1
    while True:
        t0 = time.monotonic()
        try:
            async with websockets.connect(COINBASE_WS, max_size=None, ping_interval=20) as ws:
                await ws.send(json.dumps({"type": "subscribe", "product_ids": products,
                                          "channels": ["matches", "ticker", "heartbeat"]}))
                log.info("coinbase connected: %s", products)
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "heartbeat":
                        writer.write(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("coinbase disconnected: %r", e)
        writer.write({"type": "recorder_disconnect"})
        backoff = 1 if time.monotonic() - t0 > 60 else min(backoff * 2, 60)
        await asyncio.sleep(backoff)


# ── Housekeeping ──────────────────────────────────────────────────────────────

async def monitor(writers, root, min_free_gb, stop):
    last = {w.source: 0 for w in writers}
    while not stop.is_set():
        await asyncio.sleep(60)
        free = shutil.disk_usage(root).free / 1e9
        rates = {w.source: (w.count - last[w.source]) / 60 for w in writers}
        last = {w.source: w.count for w in writers}
        size = sum(f.stat().st_size for f in Path(root).rglob("*.jsonl.gz")) / 1e9
        log.info("msgs/s %s | stored %.2f GB | disk free %.1f GB",
                 ", ".join(f"{k} {v:.0f}" for k, v in rates.items()), size, free)
        if free < min_free_gb:
            log.error("free disk below %.1f GB, stopping", min_free_gb)
            stop.set()


async def main_async(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    signer = KalshiSigner.from_env(args.env_file)
    if signer is None:
        raise SystemExit("Kalshi WebSocket needs an API key (PROD_KEYID / PROD_KEYFILE in .env)")

    ladders = dict(s.split(":") for s in args.ladder_series.split(",") if s)
    discovery = Discovery(args.series.split(","), ladders, args.ladder_hours, args.ladder_band)
    kw, cw = HourlyWriter(root, "kalshi"), HourlyWriter(root, "coinbase")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows: Ctrl-C raises KeyboardInterrupt instead
            pass

    tasks = [asyncio.create_task(KalshiRecorder(signer, discovery, kw, args.indices.split(",")).run()),
             asyncio.create_task(monitor([kw, cw], root, args.min_free_gb, stop))]
    if args.coinbase:
        tasks.append(asyncio.create_task(coinbase(cw, args.coinbase.split(","))))
    try:
        await asyncio.wait_for(stop.wait(), timeout=args.duration)
    except asyncio.TimeoutError:
        pass
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    kw.close()
    cw.close()
    log.info("stopped: %d kalshi, %d coinbase messages this run", kw.count, cw.count)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/stream")
    p.add_argument("--series", default="KXBTC15M,KXETH15M", help="15-minute series to record in full")
    p.add_argument("--ladder-series", default="KXBTCD:BRTI",
                   help="hourly ladder series and the index used to pick near-money strikes")
    p.add_argument("--ladder-hours", type=float, default=1.5, help="only ladder events closing within this many hours")
    p.add_argument("--ladder-band", type=float, default=1500, help="only strikes within this many $ of the index")
    p.add_argument("--indices", default="BRTI,ETHUSD_RTI")
    p.add_argument("--coinbase", default="BTC-USD,ETH-USD", help="Coinbase products ('' to disable)")
    p.add_argument("--min-free-gb", type=float, default=5)
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p.add_argument("--env-file", default=".env")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
