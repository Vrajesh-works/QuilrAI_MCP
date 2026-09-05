# Demo Console

A single page that drives the three HTTP gateways, so behaviour a curl dump
flattens can actually be watched.

Not part of any service. The gateways are middleware with no user-facing
surface; this is a shop window for them, and nothing depends on it.

## Run it

With the stack up (`docker compose up --build`), open **http://localhost:8000**.

Standalone, against gateways you started yourself:

```bash
uv run python -m demo_console          # :8000
```

Point it elsewhere with `CONSOLE_MCP_GATEWAY`, `CONSOLE_GUARDRAIL`,
`CONSOLE_ROUTER`, `CONSOLE_PROVIDER`, `CONSOLE_MCP_DOWNSTREAM`.

## What each panel shows

**Streaming PII redaction.** The same prompt streamed twice, side by side: raw
from the provider, and through the guardrail. Drag the chunk size down to 1 and
the redaction still lands, because the redactor holds back only the tail that
could still become a match. The console re-checks the redacted text for each
planted secret itself rather than trusting the guardrail's own report, and the
16-digit order number stays visible — it fails the Luhn check, so it is not a
card number.

**Tool authorization.** The same call under `viewer`, `admin` and no token. The
line that matters is *downstream server received* — a blocked call is answered
by the gateway itself, so the tool never runs. That comes from the downstream
server's own request log, not inferred from the error code. The tool list
includes a casing bypass (`ADMIN_reset_key`) and a batch payload that hides an
admin call behind a `tools/list`.

**Rate limiting and failover.** Force the primary to 429, hang, or fail, and
watch the request move to the backup. *Send until limited* spends the budget
until the window rejects it; the meter is the sliding window, and *New key*
switches tenant to get a fresh one.

## Notes

Serving the page here rather than opening it from disk keeps the browser on a
single origin, so the gateways need no CORS configuration.

`/api/mcp` reads the downstream server's request log through
`GET /_debug/received`, an endpoint the mock exposes for tests and this console.
A real downstream would not publish its request log.
