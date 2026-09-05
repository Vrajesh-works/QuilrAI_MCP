# MCP Security Gateway

An HTTP/JSON-RPC reverse proxy that sits between an AI agent and a downstream
MCP server. It authenticates a Bearer token, resolves the caller's role,
inspects the JSON-RPC payload, and blocks `tools/call` on `admin_*` tools for
anyone who is not an admin — returning `-32001 Unauthorized Tool Call` **without
contacting the downstream server**.

Python 3.13 · Starlette · httpx · 86 tests

## Run it

```bash
uv sync                           # from the repo root; one venv for all four
uv run python scripts/demo.py     # annotated walkthrough, in process, no ports
```

Against real ports:

```bash
uv run python -m mock_downstream   # :8081
uv run python -m mcp_gateway       # :8080

curl -s localhost:8080/mcp -H 'Authorization: Bearer viewer-token-xyz789' \
  -H 'content-type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"admin_reset_key"}}'
# {"jsonrpc":"2.0","id":1,"error":{"code":-32001,"message":"Unauthorized Tool Call",...}}
```

Demo tokens: `admin-token-abc123` (admin), `viewer-token-xyz789` (viewer).

## Request flow

```
Bearer token ──> Principal(subject, role)          401 if missing/malformed/unknown
  └─> parse JSON-RPC payload (single or batch)     -32700 / -32600 if unusable
        └─> evaluate EVERY message against policy
              ├─ blocked ──> answered here, downstream never contacted
              └─ allowed ──> forwarded
                    └─> downstream responses merged with local denials
```

The load-bearing property is *blocked calls never leave the gateway*. A proxy
that forwards first and filters the response has already let the side effect
happen — `admin_reset_key` has already rotated the key. The mock downstream
records every message it receives, and the authorization tests assert that
record stays empty.

## Behaviour

| Request | Result |
| --- | --- |
| `tools/list`, any role | Forwarded transparently (admin tools stay visible) |
| `tools/call` on a non-admin tool | Forwarded |
| `tools/call` on `admin_*`, admin role | Forwarded |
| `tools/call` on `admin_*`, other role | **`-32001` locally; downstream untouched** |
| `tools/call` with a non-string `params.name` | `-32602`, fails closed |
| `initialize`, `ping`, other methods | Forwarded — the policy governs `tools/call` only |
| Missing/malformed/unknown token | `401` + `WWW-Authenticate` |
| Unparseable body | `400` with `-32700`/`-32600`, never forwarded |
| Downstream down or slow | `502` with `-32002`, sanitised |

## Four things this gets right that are easy to miss

**1. Batches.** A gateway that reads `payload["method"]` sees nothing in
`[{tools/list}, {admin_reset_key call}]` — the array has no `method` key, the
naive check finds no admin tool, and the whole thing is forwarded. This is the
single most likely bypass. Every element is parsed and evaluated individually;
only the ones that pass are re-encoded and forwarded, and the denials are merged
into the batch response so every id is accounted for. (`tests/test_batch.py`)

**2. The prefix check is normalised before matching.** `ADMIN_reset_key`,
`" admin_reset_key"` and the fullwidth `ａdmin_reset_key` (U+FF41) all reach the
same downstream tool on a server that normalises, so all three are gated: NFKC,
strip, casefold, then compare. This is deliberately an over-approximation — it
would also gate a genuinely unprivileged `Admin_helper`. For a security filter
that is the right direction to be wrong in: a false denial is visible and
fixable, a false allow is a silent privilege escalation.

**3. A non-string tool name fails closed.** `{"name": ["admin_reset_key"]}`
makes `str.startswith` either raise or silently miss. `RpcMessage.tool_name`
returns `None` for anything that is not a plain string, and the policy refuses
what it cannot read rather than guessing.

**4. Notifications get no response, even when blocked.** JSON-RPC §4.1 is
unconditional: a message with no `id` never draws a reply. Inventing an error
frame for a blocked notification desynchronises a client that is not expecting
one, so a notification-only payload returns `204 No Content`.

## Proxy details

**The client's token stops at the gateway.** Forwarding it would let the
downstream server make its own decisions from a credential it should never see,
which defeats terminating auth here. Identity travels as `X-Forwarded-User` /
`X-Forwarded-Role` claims the gateway asserts.

**Hop-by-hop headers are stripped** (RFC 9110 §7.6.1) — `connection`,
`transfer-encoding`, `upgrade`, and friends describe one connection and must not
be relayed. `Mcp-Session-Id` *is* preserved: MCP's HTTP transport correlates
sessions with it, and a proxy that drops it silently breaks every stateful
server behind it.

**With nothing blocked, the original bytes are relayed unmodified.** Re-encoding
would normalise key order, numeric formatting and unicode escapes; a proxy that
is only inspecting should not rewrite. Re-encoding happens only when part of a
batch was filtered out.

**Upstream errors are sanitised.** `httpx.ConnectError` messages embed internal
hostnames and ports (`internal-mcp.prod.svc.cluster.local:8081`). The client
gets `-32002` and a fixed string; the operator gets the exception in the log.
Two tests assert specific internal identifiers do not appear in the response.

**Every decision is audited** on the `mcp_gateway.audit` logger with subject,
role, method, tool and outcome — an access-control gateway that cannot say who
called what is not auditable, and that log is what incident response actually
needs.

## One deviation worth flagging

`tools/list` forwards transparently by default. But
advertising `admin_reset_key` to a viewer who can never call it leaks the shape
of the privileged surface and invites the model to attempt calls that will
always fail. `MCP_GATEWAY_FILTER_TOOLS_LIST=1` strips admin tools from
`tools/list` results for non-admins. It is **off by default** so the shipped
behaviour matches the spec.

## Layout

```
src/mcp_gateway/
  app.py        routes, request flow, response merging
  jsonrpc.py    wire-format parsing (a security boundary — fails closed)
  policy.py     authorization decisions, pure functions of (principal, message)
  auth.py       Bearer token -> Principal
  proxy.py      downstream forwarding, header rules, error sanitisation
  config.py     environment configuration
src/mock_downstream/
  app.py        deliberately trusting MCP server; records what it receives
tests/
  test_authorization.py  the core rule, incl. casing/unicode bypasses
  test_batch.py          batch handling and the batch bypass
  test_authentication.py token handling, 401 semantics
  test_proxying.py       headers, transparency, upstream failure
  test_jsonrpc.py        parsing, in isolation
scripts/demo.py
```

The mock downstream performs **no authorization of its own**. That is what makes
the tests meaningful: if a privileged call reaches it, it executes — so a test
asserting a block is really asserting the gateway blocked it, not that nothing
happened to work.

## Notes and assumptions

- Tokens are a static table (`auth.py`); a real deployment verifies a signed JWT
  or calls an introspection endpoint. `resolve_principal` is the single seam.
- Auth failures never distinguish "unknown token" from "wrong role" — that would
  make the endpoint an oracle for enumerating valid tokens. Comparison is
  constant-time.
- Roles are `admin` and `viewer`; only `admin` is privileged.
- `GET` and `DELETE` on `/mcp` (the server→client stream and session teardown in
  MCP's HTTP transport) are relayed with authentication but no payload
  inspection — neither carries a JSON-RPC body to inspect. Streaming
  (`text/event-stream`) responses pass through unmodified.
- Batches that are partly blocked are re-encoded; a batch containing *any*
  unparseable member is rejected whole rather than partly forwarded.
