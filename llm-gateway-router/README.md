# LLM Gateway — Rate Limiter & Fallback Router

A resilient routing layer for an LLM gateway: a token-aware sliding window rate
limiter per tenant API key (50,000 tokens/minute, on-disk SQLite), automatic
failover to a backup provider when the primary returns 429 or exceeds its
3000ms budget, and a single standardised error shape that never leaks upstream
internals.

Python 3.13 · Starlette · httpx · stdlib `sqlite3` · 64 tests

## Run it

```bash
uv sync                           # from the repo root; one venv for all four
uv run python scripts/demo.py     # rate limiting + failover, annotated
```

Against real ports:

```bash
MOCK_NAME=primary  uv run python -m mock_model_provider   # :8101
MOCK_NAME=fallback uv run python -m mock_model_provider   # :8102
uv run python -m llm_router                         # :8100

curl -s localhost:8100/v1/messages -H 'x-api-key: sk-tenant-alpha' \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":100}'

curl -s localhost:8100/v1/usage -H 'x-api-key: sk-tenant-alpha'
```

## Request lifecycle

```
reserve quota  ->  try primary  ->  (429 / timeout / 5xx)  ->  try fallback
      |                 |                                          |
      |                 +---------------- success ----------------+
      |                                     |
      +--- release on total failure         +--> settle with real usage
```

## The sliding window

**Why not a fixed window.** "50,000 per calendar minute" lets a tenant spend the
full quota at 11:59:59 and the full quota again at 12:00:00 — 100,000 tokens in
two seconds, which is the exact burst the limit exists to prevent. This stores
individual usage events with timestamps and sums the trailing 60 seconds, so the
constraint holds at *every* instant. `test_a_fixed_window_burst_is_rejected`
pins it.

**Reserve, then settle.** Token cost is only known *after* a call completes, but
admission has to be decided *before* it starts. So each request reserves an
estimate up front (prompt estimate + the `max_tokens` ceiling it could produce),
and the reservation is corrected to the provider's reported usage on completion.

This matters more than it first appears. A limiter that only counts *completed*
requests admits any number of concurrent ones — precisely when the limit
matters. And reserving only the prompt would let a tenant sit just under the
limit and then generate an unbounded completion past it.

The flip side, which one test had to be corrected to reflect: a successful
request settles *down*, so a 20,000-token reservation that really used 150 gives
19,850 straight back. Large `max_tokens` values do not deplete a budget; real
spend does.

**Eviction is real deletion**, not just exclusion from the sum — expired rows are
deleted on every admission check, so the table stays bounded (~60 rows at one
request per second) and the hot query stays fast.

**`Retry-After` is computed, not guessed.** The limiter walks the window
oldest-first and reports when enough quota will actually have freed up, so a
client backing off correctly waits the minimum rather than a flat 60 seconds.

## Concurrency: two different mechanisms, two different scopes

This is the part I got wrong first and had to correct, so it is worth being
precise about.

**Within one process**, the admission check is made atomic by the `asyncio.Lock`
in `Store.execute`. Without it, 50 concurrent 2,000-token requests against a
50,000 limit all read "0 used" before any writes a reservation, and all 50 are
admitted. `test_concurrent_requests_cannot_exceed_the_limit` requires exactly 25.

**Across connections** — several uvicorn workers sharing one SQLite file — that
lock provides nothing, and atomicity has to come from the database. That is what
`BEGIN IMMEDIATE` is for.

The failure mode there is not what I initially assumed. Under `BEGIN DEFERRED`
both connections read the same WAL snapshot, and the second to write fails with
`SQLITE_BUSY_SNAPSHOT` (`database is locked`). Crucially **`busy_timeout` does
not retry a snapshot conflict** — so it surfaces as a hard error, and under
contention the gateway returns 500s instead of rate-limit decisions.
`BEGIN IMMEDIATE` takes the write lock *before* reading, so the second
transaction waits and then reads a snapshot that includes the first one's commit.

My first attempt at testing this didn't work: `asyncio.to_thread` dispatches
quickly enough that two checks finish one after the other and the dangerous
window never opens — the tests passed with `BEGIN DEFERRED` too, which made them
worthless as evidence. `tests/test_transactions.py` drives two connections from
threads and holds one transaction open to force the overlap. **Both tests there
fail if the mode is changed back to `BEGIN DEFERRED`**, which is what makes them
worth having.

Also tested: SQLite calls run on worker threads, so a heartbeat coroutine keeps
ticking through 100 concurrent admission checks. Blocking calls inline would
stall every in-flight request exactly when the gateway is busiest.

## Failover

| Primary result | Action | Why |
| --- | --- | --- |
| 429 | Try fallback | About *this* provider's capacity |
| Timeout (>3000ms) | Try fallback | Same |
| 5xx | Try fallback | Same |
| **4xx** | **Return it** | The same malformed body fails identically everywhere; failing over burns the backup to produce the same error twice |
| Success | Return it | Fallback never touched |

Both failing gives **502**; both timing out gives **504**, distinguishing
"nobody answered in time" from "everybody answered badly".

**On the timeout.** The 3000ms budget is enforced with `asyncio.timeout` — a
wall-clock deadline on the whole attempt — *in addition to* httpx's per-phase
`Timeout`. The outer deadline is not redundant: httpx's timeouts are per phase,
so a provider trickling one byte just inside the read timeout, forever, never
trips it and the attempt outlives its budget. Only a total deadline bounds that.

**A timeout is still charged.** It aborts *our* wait, not the provider's work —
it may still be generating, and still billing. Releasing the quota would make
timeouts free, letting a tenant drive unbounded load by always timing out. A
total failure with no timeout *does* release the reservation.

## Error sanitisation

Every client-visible failure is built in `errors.py`, so there is exactly one
place a leak could occur and one place to audit. Upstream error bodies are never
relayed — the mock providers deliberately return leaky ones:

```
"quota exhausted on deployment prod-eu-3 (tenant acct_9931, internal-llm.prod.svc)"
"internal failure at /srv/model/worker.py:412"
```

`test_errors.py` asserts each of those strings is absent from what the gateway
returns, including on the connection-error path where `httpx.ConnectError`
embeds the internal host and port.

What *is* returned is gateway-owned vocabulary — provider names, outcome
categories, elapsed times, quota numbers — which is safe by construction and
makes failures debuggable from the client side:

```json
{"type": "error",
 "error": {"type": "api_error", "message": "No model provider was able to serve this request."},
 "gateway": {"attempts": [{"provider": "primary", "outcome": "unavailable", "elapsed_ms": 1},
                          {"provider": "fallback", "outcome": "unavailable", "elapsed_ms": 1}]}}
```

The caller's tenant key is **not** forwarded upstream — it is an identity for
billing, not a credential the provider should see.

## Layout

```
src/llm_router/
  store.py      SQLite: WAL, connection handling, BEGIN IMMEDIATE
  ratelimit.py  sliding window, reserve/settle/release, eviction, retry-after
  router.py     admission, failover, settling the bill
  providers.py  one attempt, one timeout, classified into an outcome
  tokens.py     estimation before the call, reconciliation after
  errors.py     the single place client-visible errors are built
  app.py        HTTP surface
src/mock_model_provider/
  app.py        request-controlled failure modes, per-provider overrides
tests/
  test_ratelimit.py     window, eviction, reserve/settle, retry-after, restart
  test_concurrency.py   simultaneous requests, event-loop responsiveness
  test_transactions.py  forced interleaving; fails under BEGIN DEFERRED
  test_failover.py      429 / timeout / 5xx / 4xx paths and their accounting
  test_errors.py        sanitisation and the HTTP surface
scripts/demo.py
```

Tests use an injectable clock, so a 60-second window is exercised without
sleeping through it — and `time.sleep` in a concurrency test would hide the very
races those tests exist to find. The database is a real file on disk in every
test, not `:memory:`, so WAL, `busy_timeout` and file locking are actually
exercised.

## Notes and assumptions

- **On disk, not in memory.** State must survive a restart; an in-memory limiter
  hands every tenant a fresh window on each deploy, which is a trivial bypass.
  `test_state_survives_a_restart` covers it.
- Token estimation is `len(text) // 4`, deliberately rough and corrected on
  settle. A production gateway calls the provider's token-counting endpoint;
  `estimate_input_tokens` is the seam. Tool definitions are included in the
  estimate — they are frequently the largest part of a real agent request.
- A provider that reports no usage is charged the estimate. Charging zero would
  make an unreported response a free bypass.
- The tenant is the API key, from `x-api-key` or `Authorization: Bearer`.
- Fallback timeout (10s) is deliberately longer than the primary's (3s): having
  already spent the primary's budget, giving the backup the same 3s risks
  failing a request that was about to succeed.
- **Not implemented:** streaming (`stream: true`) is out of scope here — usage
  arrives in a terminal `message_delta` event, so settling would need to be
  driven from the parsed stream. The streaming guardrail in
  `llm-gateway-guardrail` has the SSE machinery this would build on.
- **Not implemented:** a circuit breaker. Every request currently pays the full
  3s primary timeout while the primary is down. Tracking consecutive failures
  and short-circuiting to the fallback for a cooldown is the obvious next step.
