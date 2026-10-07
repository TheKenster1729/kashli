# Real-time data strategy for KXBTC15M

*Last updated 2026-10-06.*

## 1. Why we are collecting tick data

The historical work (`btc15m_data.py`, `btc15m_features.py`, `btc15m_models.py`)
answered the first question: with 1-minute Kalshi candles and 1-minute Coinbase
bars, nothing beats the market's own mid price out of sample. The 15-minute
market is well calibrated at that resolution. Logistic regression and a random
forest on market, spot and context features gave no improvement in log loss or
Brier, and no trading edge after the taker fee.

The cross-market check (`btc15m_ladder.py`) showed the 15-minute window and the
hourly KXBTCD ladder settle on the same BRTI number, the 15-minute price is the
more accurate of the two, hard-bound violations after fees are rare (about 1% of
hours exceed 2 cents), and there is no taker edge in the soft deviations.

Everything left to test lives *inside* the minute: how fast the book reprices
after a spot move, whether it overshoots and comes back, which venue moves
first, and whether a resting maker order can harvest any of that. Kalshi's REST
API cannot show this and no vendor sells it, so every day the recorder is down
is data that never existed. That is the whole reason for the stream.

## 2. What is being recorded

`btc15m_stream.py` writes gzip JSON lines, one file per source per UTC hour, as
`data/stream/<source>/<YYYY-MM-DD>/<HH>.jsonl.gz`. Each line is
`{"t": <receive time, unix ms>, "m": <message>}`.

**Kalshi** (authenticated WebSocket, one connection):

| Channel | Content | Share of messages |
|---|---|---|
| `orderbook_delta` | Snapshot on subscribe, then every book change, with sequence numbers | ~92% |
| `trade` | Every fill: price, size, taker side | ~6% |
| `cfbenchmarks_value_5hz` | BRTI and ETHUSD_RTI, the indices these markets settle on, about 5 ticks/s | ~1% |
| `ticker` | Price, bid, ask, volume, open interest | ~1% |

Markets tracked: the current and next KXBTC15M and KXETH15M windows, plus
KXBTCD hourly ladder strikes within $1,500 of the live index for events closing
within 1.5 hours. About 34 tickers at any moment. Discovery runs over REST every
30 seconds and diffs into the live subscription.

**Coinbase** (public WebSocket): BTC-USD and ETH-USD `matches` (every trade) and
`ticker` (best bid and ask).

**Recorder bookkeeping messages**, written inline so gaps are visible in the data
itself: `recorder_connect` (with the market list), `recorder_disconnect`,
`recorder_markets` (adds and drops), and `recorder_gap` (an order-book sequence
gap, after which a fresh snapshot is requested).

Volume is about 0.5 GB per day when fully connected. Sequence gaps have been
rare (four on the first day, none since) so the book can be reconstructed
exactly from snapshot plus deltas.

## 3. Where the data lives

- **Windows desktop at home** (primary, since 2026-10-02). Runs `run.bat`, which
  restarts the recorder after a hard crash, registered as a startup task so it
  returns after a reboot. Sleep is disabled when plugged in. This is the machine
  that must never sleep; the MacBook run proved that a laptop cannot do the job.
- **MacBook archive** (2026-09-25 to 2026-10-02, 2.9 GB, stopped cleanly). About
  40% of those hours are usable; the rest were lost to clamshell sleep. Merge
  this tree into the Windows tree before analysis so the reader sees one
  continuous set of files. The `recorder_connect` and `recorder_disconnect`
  markers show exactly where the holes are.
- **Transfer**: share `data\stream` on the home network and pull it with rsync
  when analyzing. Keep the Windows copy as the master. A second copy on an
  external drive or cloud folder is cheap insurance at 15 GB per month.

Disk budget: 0.5 GB per day means 180 GB per year uncompressed by gzip level 6.
The recorder stops itself if free space drops below 5 GB (`--min-free-gb`), so
check free space when pulling data.

## 4. Operating checklist

Check at least weekly, or whenever the home network or power has been
disrupted:

1. `tasklist /fi "imagename eq python.exe"` shows one python process.
2. The last line of `data\stream\recorder.log` is less than two minutes old and
   reports Kalshi in the hundreds or thousands of messages per second. A Kalshi
   rate of 0 means connected to nothing; long runs of `disconnected` warnings
   mean the network is flapping.
3. The newest hourly file is the current UTC hour and still growing.
4. Free disk is comfortably above 5 GB.

Known failure modes and what they look like in the log:

- **Machine asleep**: log goes silent in 15-to-20-minute chunks with a burst of
  reconnects between them. Fix the power settings, nothing else helps.
- **Kalshi 429 rate limits**: `discovery failed: ... Too Many Requests`. Happens
  when discovery and reconnects pile up. Harmless when occasional. If it becomes
  constant, add a backoff to the discovery loop.
- **Windows Update reboot**: a few minutes missing, then a `recorder_connect`.
  Acceptable. The reader tolerates the truncated gzip tail this leaves.
- **Disk full**: `free disk below 5.0 GB, stopping` and the process exits. The
  batch loop will restart it every 30 seconds and it will stop again, so the
  log fills with this line until space is freed.

## 5. Hypotheses to test, in order

Each one is framed so the answer is a number compared against the market's own
price, after fees, with a realistic fill. That standard is non-negotiable; see
section 6.

### 5.1 Overreaction fading (maker side)

*Claim*: when BRTI jumps, the 15-minute book moves more than the fair
probability change and comes back within seconds.

Measure: for each spot move of more than some threshold within a window, record
the book mid before, the mid at its extreme in the next N seconds, and the mid
at N+ seconds. Compare the extreme to a random-walk fair probability computed
from BRTI and the time remaining. If the overshoot is systematic, simulate
resting a limit order at the fair value and ask what fraction fills and what the
expected value is after the maker fee. The fill model must use the recorded
trades: an order only fills when a trade prints through its price.

### 5.2 Lead and lag between venues

*Claim*: Coinbase trades lead BRTI, and BRTI leads the Kalshi book, by a
measurable and exploitable number of milliseconds.

Measure: cross-correlation of signed returns at 100 ms to 1 s lags across the
three series, split by time of day. If the Kalshi book lags by more than the
round trip to Kalshi, a taker order at the stale quote is profitable before fees.
Compute the edge in cents against the taker fee 0.07 * P * (1 - P). This is a
latency race, so the delayed-fill test in section 6 decides whether it is real
for us or only for someone colocated.

### 5.3 Cross-market pricing at tick resolution

*Claim*: the hourly ladder and the 15-minute window disagree transiently when
one of them has just been hit, and the other is right.

Measure: repeat the `btc15m_ladder.py` consistency bounds on the live book
rather than 1-minute candles, and record how long a bound violation persists and
how much size is on the violating quote. Violations that last under a second or
have one contract behind them are not tradeable.

### 5.4 Book shape as a signal

*Claim*: order-book imbalance (resting YES depth versus NO depth near the touch)
predicts the next mid move or the settlement better than the mid alone.

Measure: add imbalance, depth within 5 cents, and recent trade flow to the
feature table at one-second resolution and rerun the walk-forward models against
the market benchmark. This is the cheapest test once the book is reconstructed
and is a useful control: if imbalance has no value, the book is thin enough that
the earlier hypotheses are unlikely to pay either.

### 5.5 Trade-flow toxicity

*Claim*: large taker prints move the price permanently, small ones revert.

Measure: price impact at 1, 5, 30 seconds after each trade, bucketed by size and
side. Tells us whether a maker should pull quotes after a large print, which
feeds back into the fill model for 5.1.

## 6. Evaluation standard

- **Benchmark is the market, not accuracy.** Every model is scored by log loss
  and Brier against the YES mid at the same instant. Predicting the outcome from
  scratch is not the goal.
- **Fees are in.** Taker fee 0.07 * P * (1 - P) per contract. Maker fee per the
  Kalshi schedule at the time, checked for the crypto series specifically.
- **Delayed fill.** Every taker strategy is run twice: at the quote when the
  signal fires, and at the quote one second later (one minute later in the old
  work). An edge that disappears with delay is a latency race, not a forecast.
- **Maker fills are earned, not assumed.** A resting order fills only when a
  recorded trade prints at or through its price, and only for the size that
  printed.
- **Independence unit is the window.** Rows within one 15-minute market are
  correlated, so bootstrap and walk-forward splits are by market, not by row.
- **Walk-forward by week** once enough data exists, with no shuffling.

## 7. Pipeline to build

1. **Reader and book reconstruction.** Replay `orderbook_snapshot` and
   `orderbook_delta` per market into a best bid, best ask, and depth ladder at
   every tick, resetting on `recorder_gap` and `recorder_connect`. Validate
   against the `ticker` channel bid and ask.
2. **Time alignment.** Put BRTI, Coinbase trades, Kalshi trades and book
   snapshots on one receive-time axis in milliseconds. Receive time includes our
   own network latency, which is fine as long as it is treated as the moment we
   could have acted.
3. **Window table.** One row per (market, second), with book, flow, index and
   spot features, plus the settlement outcome. This is the tick-resolution
   successor to `btc15m_features.py`.
4. **Event tables** for sections 5.1, 5.2 and 5.5: spot jumps, trades, bound
   violations, each with the before and after measurements.
5. **Reports** in the same style as the existing HTML outputs.

Write the reader so it streams one hour at a time. A day is 0.5 GB compressed
and tens of millions of messages, so nothing should load a whole day into memory.

## 8. How much data is enough

- KXBTC15M alone is 96 windows per day, KXETH15M another 96.
- One week gives about 670 BTC windows, enough to see whether the overreaction
  and lead-lag effects exist at all and to size their magnitude.
- A month gives about 2,900 BTC windows, enough for a walk-forward test with
  weekly folds and for splitting by hour of day, where liquidity and behavior
  differ most.
- First analysis pass: the week of 2026-10-06, once seven clean days exist on the
  Windows machine. Do not wait for a month to look; the point of the first look
  is to find recorder or reader bugs while they are cheap to fix.

## 9. Open questions

- Does Kalshi charge a maker fee on KXBTC15M, and how much? This changes whether
  5.1 can work at all.
- How much resting size is there within a few cents of the touch in the last
  five minutes of a window? If it is a handful of contracts, the strategies here
  cannot scale and the research is for understanding rather than trading.
- Is the basic-tier key's rate limit enough for discovery plus a live trading
  loop, or will trading need its own connection?
- Should the recorder also capture SOL (KXSOL15M and SOLUSD_RTI)? The code
  supports it with a flag change. The cost is a third of the disk rate again.
