import re
import threading
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go


class BitcoinTrading:
    """
    Client for the Kalshi KXBTCD (Bitcoin daily) prediction market series.

    Primary ML-focused entry point:
        df = BitcoinTrading().collect_training_data("26FEB2500", "26MAR0423")

    Each row in the returned DataFrame represents one candlestick period
    for one market contract, with features ready for model training.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    _MONTH_MAP = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    _MONTH_INV = {v: k for k, v in _MONTH_MAP.items()}

    def __init__(self, series="KXBTCD"):
        self.series = series
        self._local = threading.local()  # thread-local HTTP session storage

    @property
    def _session(self):
        """Return a thread-local requests.Session, creating it on first access."""
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers.update({"accept": "application/json"})
            self._local.session = s
        return self._local.session

    # ── Ticker suffix helpers ────────────────────────────────────────────

    def _parse_suffix(self, suffix):
        """
        Parse a bare ``YYMMMDDH`` suffix string to a UTC-aware datetime.

        Examples:
            ``"26FEB2500"`` → datetime(2026, 2, 25, 0, tzinfo=utc)
            ``"26MAR0423"`` → datetime(2026, 3, 4, 23, tzinfo=utc)

        Raises:
            ValueError: if the suffix cannot be parsed.
        """
        match = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2})(\d{2})", suffix)
        if not match:
            raise ValueError(
                f"Cannot parse ticker suffix {suffix!r}. "
                "Expected YYMMMDDH format, e.g. '26MAR0423'."
            )
        yy, mon, dd, hh = match.groups()
        month = self._MONTH_MAP.get(mon)
        if not month:
            raise ValueError(f"Unknown month abbreviation {mon!r} in suffix {suffix!r}.")
        return datetime(2000 + int(yy), month, int(dd), int(hh), tzinfo=timezone.utc)

    def _format_suffix(self, dt):
        """
        Convert a UTC datetime to a ``YYMMMDDH`` suffix string.

        Examples:
            datetime(2026, 2, 25, 0, utc) → ``"26FEB2500"``
        """
        mon = self._MONTH_INV[dt.month]
        return f"{dt.year % 100:02d}{mon}{dt.day:02d}{dt.hour:02d}"

    def _generate_tickers(self, start, end):
        """
        Generate all hourly event tickers between ``start`` and ``end`` inclusive.

        Args:
            start (str): Start suffix in ``YYMMMDDH`` format, e.g. ``"26FEB2500"``.
            end   (str): End suffix in ``YYMMMDDH`` format, e.g. ``"26MAR0423"``.

        Returns:
            list[str]: Full tickers like ``["KXBTCD-26FEB2500", "KXBTCD-26FEB2501", ...]``.
        """
        start_dt = self._parse_suffix(start)
        end_dt   = self._parse_suffix(end)
        if start_dt > end_dt:
            raise ValueError(f"start ({start!r}) must be before or equal to end ({end!r}).")

        tickers = []
        dt = start_dt
        while dt <= end_dt:
            tickers.append(f"{self.series}-{self._format_suffix(dt)}")
            dt += timedelta(hours=1)
        return tickers

    # ── Event fetching (kept for reference / other uses) ─────────────────

    def get_events(self, status=None, min_close_ts=None):
        """Fetch events in the series, handling cursor-based pagination."""
        url = f"{self.BASE_URL}/events"
        params = {"series_ticker": self.series, "limit": 200}
        if status:
            params["status"] = status
        if min_close_ts is not None:
            params["minCloseTs"] = min_close_ts

        events = []
        first_cursor = None

        while True:
            response = self._session.get(url, params=params).json()
            events.extend(response.get("events", []))
            cursor = response.get("cursor")

            if not cursor:
                break
            if first_cursor is None:
                first_cursor = cursor
            elif cursor == first_cursor:
                break

            params["cursor"] = cursor

        return events

    def parse_event_date(self, event_ticker):
        """
        Parse the settlement date from a ticker like ``KXBTCD-25JAN0820``.

        Returns a UTC-aware :class:`datetime`, or ``None`` if parsing fails.
        """
        match = re.search(r"-(\d{2}[A-Z]{3}\d{2}\d{2})$", event_ticker)
        if not match:
            return None
        try:
            return self._parse_suffix(match.group(1))
        except ValueError:
            return None

    # ── Market / candlestick helpers ─────────────────────────────────────

    def get_markets(self, event_ticker):
        """
        Return all markets for an event, including settlement result.

        Each entry is a dict with keys:
            ticker, start_ts, end_ts, status, result
        """
        url = f"{self.BASE_URL}/markets"
        params = {"event_ticker": event_ticker}
        markets = self._session.get(url, params=params).json().get("markets", [])
        return [
            {
                "ticker":   m["ticker"],
                "start_ts": m["open_time"],
                "end_ts":   m["close_time"],
                "status":   m.get("status"),
                "result":   m.get("result"),
            }
            for m in markets
        ]

    def parse_strike_price(self, ticker):
        """
        Extract the numeric strike from a market ticker.

        Examples:
            ``KXBTCD-25JAN0820-B94000``      → 94000.0
            ``KXBTCD-25JAN1019-T101999.99``  → 101999.99
        """
        match = re.search(r"[A-Z](\d+(?:\.\d+)?)$", ticker)
        return float(match.group(1)) if match else None

    def _iso_to_epoch(self, iso_time):
        """Convert an ISO 8601 UTC string (``…Z``) to a Unix epoch float."""
        return (
            datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )

    def get_candlesticks(self, ticker, start_ts, end_ts, period_interval=1):
        """Fetch raw candlestick JSON for one market."""
        url = f"{self.BASE_URL}/series/{self.series}/markets/{ticker}/candlesticks"
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
        return self._session.get(url, params=params).json()

    # ── ML data collection ───────────────────────────────────────────────

    #: Only these three values are accepted by the Kalshi candlestick endpoints.
    VALID_INTERVALS = {1, 60, 1440}

    def collect_training_data(self, start, end, period_interval=1, max_workers=10):
        """
        Collect candlestick data for every hourly KXBTCD market between
        ``start`` and ``end`` and return a tidy :class:`pandas.DataFrame`
        ready for ML training.

        Tickers are generated directly from the ``YYMMMDDH`` format without
        querying the events API — every hour in the range is attempted in
        parallel, and hours with no markets are silently skipped.

        API calls are issued in two parallel phases via a thread pool:
          1. Fetch markets for every generated event ticker simultaneously.
          2. Fetch candlesticks for every market simultaneously.

        Args:
            start           (str): Start suffix in ``YYMMMDDH`` format,
                                   e.g. ``"26FEB2500"`` (Feb 25 2026 00:00 UTC).
            end             (str): End suffix in ``YYMMMDDH`` format,
                                   e.g. ``"26MAR0423"`` (Mar 4 2026 23:00 UTC).
            period_interval (int): Candle duration in minutes.
                                   **Must be one of 1, 60, or 1440.**
            max_workers     (int): Max concurrent API calls (default 10).

        Returns:
            pd.DataFrame: One row per (market, candle period) with columns:

            Identifiers
                event_ticker, market_ticker, event_date

            Contract metadata
                strike_price       – USD strike level for this contract
                period_interval_mins

            Temporal features  (useful as model inputs)
                period_start_ts    – Unix epoch (int)
                period_end_ts      – Unix epoch (int)
                period_start_dt    – UTC datetime
                time_to_expiry_secs – seconds until market closes
                time_elapsed_secs  – seconds since market opened
                frac_time_elapsed  – 0.0–1.0 fraction of market lifetime elapsed

            Price OHLC  (cents, 0–100 ≈ implied probability %)
                open, high, low, close

            Order-book snapshot
                yes_bid_open, yes_bid_close
                yes_ask_open, yes_ask_close

            Volume
                volume             – contracts traded in this candle

            Target variable
                settlement         – 1 (YES), 0 (NO), or NaN (not yet settled)

        Raises:
            ValueError: If ``period_interval`` is not 1, 60, or 1440.
            ValueError: If ``start`` or ``end`` cannot be parsed.
        """
        if period_interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"period_interval must be one of {sorted(self.VALID_INTERVALS)}, "
                f"got {period_interval!r}"
            )

        # ── Step 1: generate all hourly event tickers ─────────────────────
        tickers = self._generate_tickers(start, end)
        print(f"[collect] {len(tickers)} event tickers generated ({start} → {end})")

        # ── Step 2: fetch markets for all tickers concurrently ────────────
        def _fetch_markets(event_ticker):
            return event_ticker, self.get_markets(event_ticker)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            ticker_data = list(pool.map(_fetch_markets, tickers))

        event_market_pairs = [
            (event_ticker, market)
            for event_ticker, markets in ticker_data
            for market in markets
        ]
        print(f"[collect] {len(event_market_pairs)} markets total")
        if not event_market_pairs:
            return pd.DataFrame()

        # ── Step 3: fetch candlesticks for every market concurrently ──────
        def _market_rows(pair):
            event_ticker, market = pair
            event_date = self.parse_event_date(event_ticker)
            ticker      = market["ticker"]
            strike      = self.parse_strike_price(ticker)
            mkt_start   = int(self._iso_to_epoch(market["start_ts"]))
            mkt_end     = int(self._iso_to_epoch(market["end_ts"]))
            duration    = mkt_end - mkt_start

            result     = market.get("result")
            settlement = 1 if result == "yes" else (0 if result == "no" else None)

            candles = self.get_candlesticks(
                ticker, mkt_start, mkt_end, period_interval
            ).get("candlesticks", [])

            rows = []
            for candle in candles:
                end_ts   = candle["end_period_ts"]
                start_ts = end_ts - period_interval * 60
                price    = candle.get("price", {})
                yes_bid  = candle.get("yes_bid", {})
                yes_ask  = candle.get("yes_ask", {})
                rows.append(
                    {
                        # ── Identifiers ──────────────────────────
                        "event_ticker":         event_ticker,
                        "market_ticker":        ticker,
                        "event_date":           event_date,
                        # ── Contract metadata ─────────────────────
                        "strike_price":         strike,
                        "period_interval_mins": period_interval,
                        # ── Temporal features ─────────────────────
                        "period_start_ts":      start_ts,
                        "period_end_ts":        end_ts,
                        "period_start_dt":      datetime.fromtimestamp(start_ts, tz=timezone.utc),
                        "time_to_expiry_secs":  mkt_end - end_ts,
                        "time_elapsed_secs":    start_ts - mkt_start,
                        "frac_time_elapsed":    (start_ts - mkt_start) / duration if duration > 0 else None,
                        # ── Price OHLC (trade prices; None when no trades) ──
                        "open":                 price.get("open_dollars"),
                        "high":                 price.get("high_dollars"),
                        "low":                  price.get("low_dollars"),
                        "close":                price.get("close_dollars"),
                        # ── Order-book ────────────────────────────
                        "yes_bid_open":         yes_bid.get("open_dollars"),
                        "yes_bid_close":        yes_bid.get("close_dollars"),
                        "yes_ask_open":         yes_ask.get("open_dollars"),
                        "yes_ask_close":        yes_ask.get("close_dollars"),
                        # ── Volume ────────────────────────────────
                        "volume":               candle.get("volume"),
                        # ── Target variable ───────────────────────
                        "settlement":           settlement,
                    }
                )
            return rows

        all_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for rows in pool.map(_market_rows, event_market_pairs):
                all_rows.extend(rows)

        print(f"[collect] {len(all_rows)} candlestick rows collected")
        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df = df.sort_values(
            ["event_ticker", "market_ticker", "period_start_ts"]
        ).reset_index(drop=True)
        return df

    def to_csv(self, df, filepath):
        """Persist the training DataFrame to *filepath* as CSV."""
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} rows to {filepath}")

    # ── Visualisation ────────────────────────────────────────────────────

    def convert_ts_to_datetime(self, iso_time):
        dt = datetime.strptime(iso_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return dt.timestamp()

    def get_start_and_end_ts(self, market):
        return (
            self.convert_ts_to_datetime(market["start_ts"]),
            self.convert_ts_to_datetime(market["end_ts"]),
        )

    def plot_candlesticks(self, candlesticks, price, fig):
        times, opens, highs, lows, closes = [], [], [], [], []
        for candle in candlesticks:
            p = candle.get("price", {})
            o = p.get("open_dollars")
            h = p.get("high_dollars")
            l = p.get("low_dollars")
            c = p.get("close_dollars")
            # Fall back to yes_bid/yes_ask midpoint when there are no trades
            if None in (o, h, l, c):
                bid = candle.get("yes_bid", {})
                ask = candle.get("yes_ask", {})
                def _mid(key):
                    b = bid.get(key)
                    a = ask.get(key)
                    if b is None or a is None:
                        return None
                    return (float(b) + float(a)) / 2
                o, h, l, c = _mid("open_dollars"), _mid("high_dollars"), _mid("low_dollars"), _mid("close_dollars")
            if None in (o, h, l, c):
                continue
            end_time   = datetime.fromtimestamp(candle["end_period_ts"])
            start_time = end_time - timedelta(hours=1)
            times.append(start_time)
            opens.append(float(o))
            highs.append(float(h))
            lows.append(float(l))
            closes.append(float(c))
        fig.add_trace(
            go.Candlestick(
                x=times, open=opens, high=highs, low=lows, close=closes,
                increasing_line_color="green", decreasing_line_color="red",
                name=f"Price: {price}",
            )
        )

    def get_candlesticks_for_event(self, event_ticker, period_interval=1):
        markets = self.get_markets(event_ticker)
        fig = go.Figure()
        for market in markets:
            if not market:
                continue
            ticker = market["ticker"]
            try:
                price = float(ticker.split("-")[-1])
            except ValueError:
                price = float(ticker.split("-")[-1][1:])
            start_ts, end_ts = self.get_start_and_end_ts(market)
            data = self.get_candlesticks(ticker, int(start_ts), int(end_ts), period_interval)
            self.plot_candlesticks(data.get("candlesticks", []), price, fig)
        return fig

# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bt = BitcoinTrading()

    # Collect hourly candles from Feb 25 2026 00:00 UTC to Mar 4 2026 23:00 UTC.
    # Valid period_interval values: 1 (1 min), 60 (1 hour), 1440 (1 day).
    # df = bt.collect_training_data("26FEB2500", "26MAR0423", period_interval=60)
    # print(df.shape)
    # print(df["event_ticker"].nunique(), "unique events")
    # print(df.head())
    fig = bt.get_candlesticks_for_event("KXBTCD-26MAR0712")
    fig.show()