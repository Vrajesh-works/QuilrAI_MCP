# QuilrAI_MCP

Model Context Protocol (MCP) servers, MCP gateways, LLM gateways, security
guardrails, and system integration.

Four services that sit between AI agents and the models and tools they call.
Python 3.13 throughout, organised as a single `uv` workspace: one shared
environment, four independently runnable projects.

| Directory | What it is |
| --- | --- |
| [`mcp-server/`](./mcp-server) | stdio MCP server — customer lookup and refunds, strict validation, stdout purity |
| [`mcp-gateway/`](./mcp-gateway) | HTTP/JSON-RPC security gateway — bearer auth, `admin_*` tool authorization |
| [`llm-gateway-guardrail/`](./llm-gateway-guardrail) | LLM gateway — real-time streaming PII redaction |
| [`llm-gateway-router/`](./llm-gateway-router) | LLM gateway — token-aware rate limiting and model failover |
| [`demo-console/`](./demo-console) | Browser console for driving the three gateways; not part of any service |

## Running it

### Docker

The three gateways, their mock upstreams and the console, wired together:

```bash
docker compose up --build
```

Then open **http://localhost:8000** for the console, or use the ports directly:

| Service | Port | |
| --- | --- | --- |
| `console` | 8000 | browser UI for all three gateways |
| `mcp-gateway` | 8080 | + `mcp-downstream` (internal) |
| `llm-guardrail` | 8090 | + `llm-provider` (internal) |
| `llm-router` | 8100 | + `model-primary`, `model-fallback` (internal) |

```bash
# A viewer calling an admin tool is blocked by the gateway itself
curl -s localhost:8080/mcp   -H 'Authorization: Bearer viewer-token-xyz789'   -H 'content-type: application/json'   -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}'
# {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call",...}}

# PII redacted despite arriving three characters at a time
curl -sN localhost:8090/v1/messages   -H 'content-type: application/json'   -d '{"stream":true,"chunk_size":3,"text":"ssn is 123-45-6789 done"}'
# ...{"text":"[REDACTED] "}...

# Primary returns 429, so the request moves to the backup
curl -s -D- -o /dev/null localhost:8100/v1/messages   -H 'x-api-key: sk-tenant-alpha'   -H 'content-type: application/json'   -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":100,"behaviour":"rate_limited","fallback_behaviour":"ok"}'
# x-gateway-provider: fallback
```

`docker compose down -v` stops everything and drops the router's volume. To run
one stack on its own, `docker compose up llm-router` pulls in only the providers
it depends on.

One image serves all eight services; they differ only by the command compose
gives them. The router's SQLite window lives on a named volume, so it survives
`docker compose restart` — the property `test_state_survives_a_restart` asserts.

The MCP server is not containerised: it speaks stdio, so it has no port to
publish. Run it through `claude mcp add` as its README describes, or see the
transcript with `uv run python mcp-server/scripts/probe.py`.

### uv

For tests and development. A
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): one
shared `.venv` and lockfile at the root, each service still its own project.

```bash
uv sync
uv run python run_tests.py
```

The demo scripts print annotated transcripts and start nothing:

```bash
uv run python mcp-server/scripts/probe.py             # JSON-RPC transcript + stderr proof
uv run python mcp-gateway/scripts/demo.py             # blocked calls never reach downstream
uv run python llm-gateway-guardrail/scripts/demo.py   # PII redacted at 1/5/40 chars per chunk
uv run python llm-gateway-router/scripts/demo.py      # failover, budget exhaustion, concurrency
```

Working on one service in isolation:

```bash
cd mcp-gateway && uv run pytest -q
uv run python run_tests.py llm-gateway-router
```

`run_tests.py` runs each project in its own process rather than calling `pytest`
once at the root: all four have a `tests/conftest.py` and each puts its own
`tests/` on `sys.path`, so in a single process those four `conftest` modules
collide and three suites fail to import.

Each directory has its own README covering design decisions, assumptions, and
how to run it standalone.


## mcp-server

An MCP server over stdio exposing `get_customer_record` and `trigger_refund`,
built on the SDK's low-level `Server` so invalid input produces real JSON-RPC
error codes rather than `isError` results.

The design decision worth reading: **two error channels**. A schema violation is
`-32602` (the caller broke the contract); "customer not found" or "insufficient
refundable balance" is a successful result with `isError: true`, so the model
can read the outcome and adjust instead of seeing a transport failure.

Stdout purity is tested against a real subprocess that deliberately misbehaves —
`print()` to stdout, raw writes, warnings, a chatty dependency logger — and the
wire still frames cleanly.

## mcp-gateway

A reverse proxy between an agent and a downstream MCP server. Authenticates a
Bearer token, resolves the role, and blocks `tools/call` on `admin_*` for
non-admins with `-32001` **without contacting the downstream server**.

Batch payloads are inspected element by element (the obvious
bypass), the `admin_` prefix check normalises Unicode and case before matching,
non-string tool names fail closed, and blocked notifications correctly draw no
response.

## llm-gateway-guardrail

A proxy that redacts PII from LLM responses **as they stream**, without buffering
the response. The hard part is that matches straddle chunk boundaries: a model
sending `"...ssn is 123-4"` then `"5-6789..."` defeats per-chunk regex entirely.

The fix is a **holdback** — retain only the shortest suffix that could still
become a match (found with real partial matching), and emit everything before it
immediately. Ordinary prose holds back nothing. Measured at 1 character per
delta: TTFT 94ms of a 4.2s stream, peak holdback 27 characters, byte-identical
output across every chunk size.

Two cases produce silent failures rather than crashes, and both are covered by
tests: a *complete* match at the buffer's end can still grow (`[REDACTED]m`),
and one pattern's partial match can land inside another's complete match,
splitting a phone number in half.

## llm-gateway-router

Token-aware **sliding window** rate limiting (50,000 tokens/min per tenant key,
on-disk SQLite) plus automatic failover when the primary returns 429 or blows
its 3000ms budget.

Quota is **reserved before** a call on an estimate and **settled after** against
the provider's reported usage — a limiter that only counts completed requests
admits unlimited concurrent ones, which is exactly when the limit matters.

Two distinct concurrency mechanisms, at different scopes: an `asyncio.Lock`
makes the admission check atomic between coroutines, while `BEGIN IMMEDIATE`
handles multiple workers sharing the SQLite file. Under `BEGIN DEFERRED` the
second writer hits `SQLITE_BUSY_SNAPSHOT`, which `busy_timeout` does *not*
retry — a hard error rather than a wait. `test_transactions.py` forces that
interleaving and fails if the mode is changed back.
