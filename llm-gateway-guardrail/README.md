# LLM Gateway — Streaming PII Guardrail

A proxy that relays text generation to an LLM provider and redacts PII from the
response **as it streams**, without buffering the response or noticeably
delaying the first token.

Python 3.13 · Starlette · httpx · `regex` · 117 tests

## Run it

```bash
uv sync                           # from the repo root; one venv for all four
uv run python scripts/demo.py     # watch a leaky response get redacted
```

Against real ports:

```bash
uv run python -m mock_provider    # :8091
uv run python -m llm_guardrail    # :8090

curl -N localhost:8090/v1/messages -H 'content-type: application/json' \
  -d '{"stream":true,"chunk_size":3,"text":"ssn 123-45-6789 ok"}'
```

Point it at a real provider with `LLM_UPSTREAM_URL`; the caller's credentials
are forwarded untouched, so this is a drop-in path to the Messages API.

## The problem

A stream delivers text in arbitrary pieces. A model emitting an SSN might send:

```
chunk 1: "...my ssn is 123-4"
chunk 2: "5-6789..."
```

Running a regex over each chunk independently finds nothing in either, and the
SSN goes out intact. **Matches straddle chunk boundaries** — that is the whole
difficulty of the problem.

The obvious fix — buffer the whole response, redact once, send — is correct and
useless: it converts time-to-first-token into time-to-*last*-token and grows
memory with response length.

## The approach

Keep a **holdback**: the shortest suffix of pending text that could still turn
into a match. Everything before it can never be part of a future match, so it is
safe to emit immediately.

```
feed("Your ssn is 123-4")  ->  emits "Your ssn is ",  holds "123-4"
feed("5-6789 ok")          ->  emits "[REDACTED] ok", holds ""
```

Finding that point needs a real partial-match engine, not a heuristic. The
`regex` module's `partial=True` reports "the pattern ran out of input mid-match",
which is exactly the question being asked. For ordinary prose the holdback is
empty and text flows straight through.

**Measured** (`scripts/demo.py`, worst case of 1 character per delta):

| | |
| --- | --- |
| TTFT | 94ms of a 4,201ms stream |
| Peak holdback | 27 characters |
| Output at chunk sizes 1 / 5 / 40 | byte-identical |

## Two bugs this design has to get right

Both were found by tests in this suite, and both produce *silent* leaks or
corruption rather than crashes.

**1. A complete match can still grow.** `ada.lovelace@example.co` is a complete,
valid email match. Redact it the moment it appears and the `m` arriving in the
next chunk lands *after* the placeholder:

```
[REDACTED]m
```

So a match ending at the very end of the buffer is held, not redacted, until
something follows it. Only matches that can no longer change are rewritten.

**2. One pattern's partial match can land inside another's complete match.**
Given `value (555) 123-4567 `, the credit-card pattern reports a partial match
starting at `123-4567` — in the middle of the phone number. Taking that as the
holdback point splits the phone number in half, neither half matches, and the
whole thing is emitted in the clear.

The fix is ordering: redact settled matches *first*, then compute the holdback on
what remains. That removes the text the spurious partial was feeding on. The
comment in `feed()` says so, because the two lines look trivially reorderable and
are not.

## Correctness testing

The oracle is whole-text redaction: **streaming output must equal non-streaming
output, for every possible chunking.**

- `test_split_at_every_single_offset` breaks each secret at *every* offset in
  turn — boundary bugs hide at one specific position, like just after the `@`.
- `test_redaction_survives_every_chunk_size` sweeps chunk sizes 1…64.
- `test_randomised_chunkings_agree_with_whole_text_redaction` runs 200 seeded
  random chunkings.
- `test_raw_bytes_on_the_wire_contain_no_pii` asserts on the bytes the client
  receives, not the reassembled text, so a leak cannot hide in an event the
  reassembler ignores.

## Latency and memory testing

A redactor that buffers everything passes every correctness test above and is
still useless, so those properties get their own tests:

- TTFT must be a small fraction of total stream time, and must **not grow with
  response length** — the signature of buffer-then-redact.
- Peak holdback stays under 64 characters across a 200,000-character stream.
- An adversarial all-digit stream (ask the model to count) cannot grow the
  buffer: without a ceiling that is an unbounded allocation, since it looks like
  a forever-growing partial card number.
- 8× the input takes well under 8× the time, ruling out the rescan-everything
  implementation that is correct but quadratic.

One note on the harness: httpx's `ASGITransport` accumulates the entire response
body before returning it, so it **cannot** measure TTFT. The latency tests
therefore drive the stream transformer directly, plus one test
(`test_the_deployed_http_path_streams_too`) that runs the guardrail and provider
on real sockets to prove the deployed HTTP path streams as well.

## What is detected

| Pattern | Notes |
| --- | --- |
| Email | RFC-shaped, not RFC-complete; quoted-string forms are not worth the false positives |
| SSN | Excludes never-issued ranges (`000`/`666`/`9xx` areas, `00` group, `0000` serial) |
| Credit card | 13–19 digits, space/hyphen grouped, **validated with Luhn** |
| Phone | North American formats, including `(555) 123-4567` |
| API keys | `sk-ant-…`, `sk-…`, `AKIA…` — leaking one is as bad as leaking PII |

**Luhn is a deliberate trade.** Without it, every 16-digit order or tracking
number is redacted — a false positive users notice immediately, which erodes
trust in the guardrail. A 16-digit string that fails Luhn is not a card number,
so letting it through is not a leak. `test_non_luhn_sixteen_digit_number_is_left_alone`
pins this.

Every pattern is anchored on a word boundary and has a declared `max_length`.
Both are load-bearing: the anchor makes safe emit points fall on token
boundaries, and the length bounds the holdback ceiling.

## Layout

```
src/llm_guardrail/
  redactor.py   the core algorithm — holdback, partial matching, bounded memory
  patterns.py   detectors, validators, and the max-length bounds
  sse.py        incremental SSE parsing (same boundary problem, one layer down)
  stream.py     applies the redactor to a live SSE stream
  app.py        the proxy endpoint
src/mock_provider/
  app.py        Anthropic-shaped SSE with controllable text, chunking and delay
tests/
  test_redactor.py   the algorithm, incl. exhaustive split-offset sweeps
  test_sse.py        byte-at-a-time SSE parsing
  test_streaming.py  end-to-end through the proxy
  test_latency.py    TTFT, bounded memory, non-quadratic throughput
scripts/demo.py
```

## Notes and assumptions

- Only `text_delta` events are rewritten. `thinking_delta` and `input_json_delta`
  share the event name but are not response text; rewriting them corrupts the block.
- The redactor is flushed at `content_block_stop` and again at end of stream, so
  a response ending mid-partial-match does not lose its tail.
- A delta that redacts to nothing is dropped rather than sent as an empty event.
- Unlike the MCP gateway, this proxy does **not** terminate authentication — it is
  a transparent path to the provider and forwards the caller's credential.
- `stream: false` requests are redacted too, via the same patterns.
- Upstream failures return a sanitised `502`; internal hostnames and ports never
  reach the client.
- **Redaction is one-way.** There is no unredact path and no vault — if the
  application needs to recover the original values, this needs tokenisation
  (reversible, keyed placeholders) rather than replacement.
- Request bodies are **not** scanned. Only the response is guarded; prompt-side
  PII is a separate control.
