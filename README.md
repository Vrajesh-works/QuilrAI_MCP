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

## Running it

A [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): one
shared `.venv` and lockfile at the root, each service still its own project.

```bash
uv sync
uv run python run_tests.py
```

See each one actually work — annotated transcripts, no ports needed:

```bash
uv run python mcp-server/scripts/probe.py             # JSON-RPC transcript + stderr proof
uv run python mcp-gateway/scripts/demo.py             # blocked calls never reach downstream
uv run python llm-gateway-guardrail/scripts/demo.py   # PII redacted at 1/5/40 chars per chunk
uv run python llm-gateway-router/scripts/demo.py      # failover, budget exhaustion, concurrency
```

Working on one service in isolation still works as before:

```bash
cd mcp-gateway && uv run pytest -q
uv run python run_tests.py llm-gateway-router
```

`run_tests.py` runs each project in its own process rather than calling `pytest`
once at the root: all four have a `tests/conftest.py` and each puts its own
`tests/` on `sys.path`, so in a single process those four `conftest` modules
collide and three suites fail to import.

### Docker

The demo scripts above need no setup at all, but running the gateways *wired to
their upstreams* means seven processes. Compose collapses that to one command:

```bash
docker compose up --build
```

| Service | Port | |
| --- | --- | --- |
| `mcp-gateway` | 8080 | + `mcp-downstream` (internal) |
| `llm-guardrail` | 8090 | + `llm-provider` (internal) |
| `llm-router` | 8100 | + `model-primary`, `model-fallback` (internal) |

```bash
curl -s localhost:8080/mcp -H 'Authorization: Bearer viewer-token-xyz789'   -H 'content-type: application/json'   -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}'
# {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call",...}}

curl -sN localhost:8090/v1/messages -H 'content-type: application/json'   -d '{"stream":true,"chunk_size":3,"text":"ssn is 123-45-6789 done"}'
# ...{"text":"[REDACTED] "}...  redacted despite arriving 3 characters at a time

curl -s -D- -o /dev/null localhost:8100/v1/messages -H 'x-api-key: sk-tenant-alpha'   -H 'content-type: application/json'   -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":100,
       "behaviour":"rate_limited","fallback_behaviour":"ok"}'
# x-gateway-provider: fallback     <- primary returned 429, failed over
```

**One image, seven services.** The workspace pays off here: a single `uv sync`
installs all seven packages, so the services differ only by the command compose
gives them. Per-project images would rebuild an almost identical dependency set
four times.

**The MCP server is deliberately not containerised.** It speaks stdio, so it has
no port to publish; a container would only complicate the `claude mcp add`
invocation in its README.

The router's SQLite window lives on a named volume, so it survives
`docker compose restart` — the same property `test_state_survives_a_restart`
asserts. `uv` remains the primary path for tests and development; Docker is for
seeing the wired-up system.

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

Two silent-failure bugs the design has to handle, both caught by tests: a
*complete* match at the buffer's end can still grow (`[REDACTED]m`), and one
pattern's partial match can land inside another's complete match, splitting a
phone number in half.

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
