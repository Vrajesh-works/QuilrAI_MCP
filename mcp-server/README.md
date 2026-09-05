# MCP Server — Customer Billing

An MCP server over stdio exposing two tools, `get_customer_record` and
`trigger_refund`, with strict input validation and a hard guarantee that stdout
carries nothing but JSON-RPC.

Python 3.13 · [`mcp`](https://pypi.org/project/mcp/) 2.1 · Pydantic 2

## Run it

```bash
uv sync                           # from the repo root; one venv for all four
uv run python scripts/probe.py    # annotated request/response transcript
uv run python -m customer_mcp     # speaks MCP on stdio; expects a client on the other end
```

Register it with a client:

```bash
claude mcp add customer-mcp -- uv --directory /path/to/mcp-server run python -m customer_mcp
```

Set `CUSTOMER_MCP_LOG_LEVEL=DEBUG` for verbose logs (on stderr, always).

## The two error channels

The main design decision here. A schema violation and a business refusal are
different kinds of failure and go back to the caller differently:

| Situation | Channel | Code |
| --- | --- | --- |
| Bad `customer_id` format, non-positive/non-finite/sub-cent `amount`, `reason` under 10 chars, wrong type, missing field, unexpected extra field | JSON-RPC **error** | `-32602` Invalid params |
| Unknown tool name | JSON-RPC **error** | `-32601` Method not found |
| Malformed JSON on the wire | JSON-RPC **error** | `-32700` / `-32600` (SDK) |
| Customer not found, account frozen or closed, amount exceeds refundable balance | **Result** with `isError: true` | — |
| Unexpected internal exception | JSON-RPC **error** | `-32603`, message sanitised |

The reasoning: `-32602` means *the caller broke the contract* — the request was
never executable. But "CUST-55555 does not exist" is a valid, well-formed
question with a real answer. Returning that as a protocol error hides it from
the model as a transport failure; returning it as `isError: true` puts it in
context where the model can read it and adjust. Collapsing both into one channel
is the easy mistake, in either direction.

Validation errors name every offending field at once:

```json
{"code": -32602,
 "message": "Invalid arguments for tool 'trigger_refund': amount: Input should be greater than 0; reason: ...",
 "data": {"tool": "trigger_refund", "issues": [
   {"field": "amount", "error": "Input should be greater than 0", "type": "greater_than"},
   {"field": "reason", "error": "must be at least 10 characters of actual text (got 4 after trimming whitespace)", "type": "value_error"}]}}
```

Caller-supplied *values* are never echoed back into the payload — only field
names, reasons and error types. Arguments can hold real customer data, and error
payloads propagate into client logs and model context.
(`test_error_payload_does_not_echo_caller_input`)

## STDIO isolation

Two independent mechanisms, and one deliberate non-mechanism:

1. **`configure_logging()` runs before anything else** in `__main__.py`, binding
   every log record and captured warning to stderr from the first line. It also
   strips any handler a dependency installed at import time, since we cannot
   assume someone else's handler points at stderr.

2. **The SDK claims the descriptors.** In `mcp` ≥ 2.x, `stdio_server()` points
   fd 1 at stderr for the life of the session and serves the wire from a private
   duplicate. A stray `print()` — ours, a dependency's, or a child process's —
   lands on stderr and cannot corrupt a frame.

3. **We deliberately do *not* dup2 fd 1 ourselves.** The obvious hardening
   (`os.dup2(2, 1)` at startup, keeping a private copy for the transport) is
   actively harmful against this SDK version: doing it before `stdio_server()`
   starts would leave the transport writing JSON-RPC onto *stderr* — precisely
   the corruption the redirect exists to prevent. Verified by reading
   `mcp/server/stdio.py` rather than assumed.

Because the guarantee is about a file descriptor, it is tested against a real
process rather than in-process. `tests/test_stdout_purity.py` spawns
`python -m customer_mcp`, runs a full session, and asserts every non-empty
stdout line parses as a JSON-RPC object with `"jsonrpc": "2.0"`.

The load-bearing case is `test_stdout_stays_pure_despite_a_noisy_dependency`:
`CUSTOMER_MCP_DEMO_NOISE=1` makes the server deliberately `print()` to stdout,
write raw bytes to `sys.stdout`, emit a `warnings.warn`, and log from a
third-party-style logger, mid-session. The test asserts that noise appears in
stderr (otherwise it proves nothing) and that stdout still frames cleanly. Clean
framing from well-behaved code only shows nobody happened to call `print()`.

## Validation rules

| Field | Rule | Notes |
| --- | --- | --- |
| `customer_id` | `^CUST-\d{5}$` | Anchored: `cust-00001`, `CUST-0001`, `CUST-000001`, `" CUST-00001"` all rejected |
| `amount` | `0 < amount ≤ 100,000.00`, finite, ≤ 2 decimals | Bool and string rejected explicitly |
| `reason` | ≥ 10 chars **after stripping** | Stored stripped |
| *(any)* | `extra="forbid"` | A typo'd field name is a loud error, not a silent drop |

Three of these are worth calling out:

- **`True` is an `int` in Python.** Without an explicit bool guard, `amount: true`
  validates as `1.0` and a boolean becomes a one-dollar refund.
- **A bare `min_length=10` on `reason` accepts twelve spaces.** Length is measured
  after stripping, so it does not.
- **An integer `amount` is accepted.** JSON has no int/float distinction, so
  `100` means $100. Pydantic's `strict=True` would reject it — correct by the
  type system, wrong for the protocol.

`allow_inf_nan=False` states the finiteness requirement outright. The `gt`/`le`
bounds happen to reject NaN too, but only incidentally: every comparison against
NaN is False, so that is the bounds check failing closed rather than an actual
finiteness test.

## Layout

```
src/customer_mcp/
  __main__.py       entrypoint; logging first, then stdio transport
  server.py         tool declarations, dispatch, error-channel routing
  schemas.py        Pydantic models — the single source of truth for schemas
  errors.py         exception -> JSON-RPC error mapping, sanitisation
  store.py          in-memory customers and refund ledger
  logging_setup.py  stderr-only logging
tests/
  test_validation.py     26 malformed inputs -> -32602, in-process
  test_tools.py          discovery, happy paths, domain refusals
  test_stdout_purity.py  real subprocess; the fd-level guarantee
scripts/probe.py    manual JSON-RPC transcript
```

`tools/list` advertises `Model.model_json_schema()` from the same models that
`tools/call` validates against, so the advertised schema cannot drift from the
enforced one — `test_list_tools_advertises_both_tools_with_generated_schemas`
asserts they are equal.

Built on the low-level `Server` rather than `FastMCP` deliberately: FastMCP
converts a handler exception into a `CallToolResult` with `isError: true` — a
successful *response*. Invalid input must produce JSON-RPC *error codes*, which requires
raising `MCPError` and letting the dispatcher serialise it.

## Assumptions

- `CUST-XXXXX` is exactly five digits; no lowercase, no surrounding whitespace.
- `amount` is USD, capped at 100,000.00 as a fat-finger guard.
- Data is in-memory fixtures, reset on restart. Swapping in a real store means
  replacing `store.py`; nothing above it depends on the storage choice. `CUST-00042` is active with a 320.00 refundable balance,
  `CUST-00007` has 0.00, `CUST-01337` is frozen, `CUST-99999` is closed.
- Refunding the balance exactly to zero is allowed; the check is `>`, not `>=`.

## One SDK behaviour worth knowing

Writing every request to the server's stdin and immediately closing it can lose
the last responses: on EOF the writer task can be torn down with frames still
queued. Real clients hold stdin open for the session, so this does not arise in
normal use, but it does make `subprocess.run(input=...)` an unreliable way to
test a stdio MCP server — the first version of the purity test lost two
responses that way. `run_server()` in `tests/test_stdout_purity.py` keeps stdin
open until the expected responses arrive, which is what a real client does.
