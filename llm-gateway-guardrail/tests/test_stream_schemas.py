"""Every response-stream schema, with an explicit policy and a leak assertion.

The audit found five distinct channels through which PII reached the client
untouched, and none of them were covered by a test. Worse, one of them had a
*passing* test asserting the leak was correct. This file is the fail-closed
regression surface: for every shape the gateway might see, it states the policy
and then checks the blunt property that matters - the sensitive value does not
appear in the bytes sent to the client.

Nothing here compares against `redact_text`; the secrets are literals and the
assertions are about their absence.
"""

from __future__ import annotations

import json

import pytest
from llm_guardrail.redactor import StreamRedactor
from llm_guardrail.stream import StreamPolicy, redact_sse_stream

pytestmark = pytest.mark.asyncio

SSN = "123-45-6789"
EMAIL = "bob@secret.com"
CARD = "4111111111111111"
SECRETS = (SSN, EMAIL, CARD)


def sse(payload: dict, event: str | None = None) -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(payload)}\n\n".encode()


async def run(frames: list[bytes], policy: StreamPolicy | None = None, chunk: int | None = None) -> bytes:
    """Drive the stream, optionally re-cutting the bytes at `chunk` size."""
    raw = b"".join(frames)
    pieces = [raw[i : i + chunk] for i in range(0, len(raw), chunk)] if chunk else frames

    async def upstream():
        for piece in pieces:
            yield piece

    out = b""
    async for produced in redact_sse_stream(upstream(), StreamRedactor(), policy):
        out += produced
    return out


def assert_no_secrets(out: bytes, secrets=SECRETS) -> None:
    for secret in secrets:
        assert secret.encode() not in out, f"{secret!r} leaked: {out[:400]!r}"


# --------------------------------------------------------------------------
# The five confirmed leak channels.
# --------------------------------------------------------------------------


async def test_openai_shaped_delta_content_is_redacted():
    """vLLM, Together, Groq, LiteLLM and Azure OpenAI all speak this shape."""
    out = await run([sse({"choices": [{"index": 0, "delta": {"content": f"email {EMAIL} ssn {SSN}"}}]})])
    assert_no_secrets(out)
    assert b"[REDACTED]" in out


async def test_openai_shaped_content_split_across_two_events_is_redacted():
    """Stateful, not just a per-event scan."""
    out = await run(
        [
            sse({"choices": [{"index": 0, "delta": {"content": "my ssn is 123-4"}}]}),
            sse({"choices": [{"index": 0, "delta": {"content": "5-6789 ok"}}]}),
        ]
    )
    assert_no_secrets(out)
    assert b"[REDACTED]" in out


async def test_thinking_delta_does_not_reach_the_client():
    out = await run([sse({"delta": {"type": "thinking_delta", "thinking": f"{EMAIL} and {SSN}"}})])
    assert_no_secrets(out)


async def test_thinking_delta_is_redacted_when_policy_says_so():
    out = await run(
        [sse({"delta": {"type": "thinking_delta", "thinking": f"{EMAIL} and {SSN}"}})],
        policy=StreamPolicy(thinking="redact"),
    )
    assert_no_secrets(out)
    assert b"thinking_delta" in out, "redact policy keeps the block, minus the PII"


async def test_thinking_delta_passes_only_when_explicitly_configured():
    """`pass` must be reachable only by deliberate configuration."""
    out = await run(
        [sse({"delta": {"type": "thinking_delta", "thinking": SSN}})],
        policy=StreamPolicy(thinking="pass"),
    )
    assert SSN.encode() in out, "the escape hatch should still work when asked for"


async def test_tool_call_arguments_are_redacted():
    """`input_json_delta` carries tool arguments, which is where identifiers live."""
    out = await run(
        [
            sse({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"ssn":"123-45-'}}),
            sse({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '6789"}'}}),
        ]
    )
    assert_no_secrets(out)
    assert b"[REDACTED]" in out


async def test_message_start_content_is_redacted():
    out = await run(
        [
            sse(
                {
                    "type": "message_start",
                    "message": {"id": "msg_1", "content": [{"type": "text", "text": f"ssn {SSN}"}]},
                },
                event="message_start",
            )
        ]
    )
    assert_no_secrets(out)


async def test_an_unknown_delta_type_is_redacted_not_relayed():
    """The channel that does not exist yet. This is the one that matters most,
    because it is the only one that cannot be enumerated in advance."""
    out = await run([sse({"delta": {"type": "output_text_delta", "text": f"ssn {SSN}"}})])
    assert_no_secrets(out)


async def test_an_entirely_unknown_envelope_is_redacted():
    out = await run([sse({"some_future_field": {"nested": [{"deep": f"card {CARD}"}]}})])
    assert_no_secrets(out)


async def test_unknown_schema_can_be_blocked_outright():
    out = await run(
        [sse({"mystery": f"ssn {SSN}"})],
        policy=StreamPolicy(unknown_schema="block"),
    )
    assert out == b""


async def test_unknown_schema_passes_only_when_explicitly_configured():
    out = await run([sse({"mystery": f"ssn {SSN}"})], policy=StreamPolicy(unknown_schema="pass"))
    assert SSN.encode() in out


# --------------------------------------------------------------------------
# Malformed and hostile framing.
# --------------------------------------------------------------------------


async def test_malformed_json_frames_do_not_leak_and_do_not_kill_the_stream():
    out = await run(
        [
            b"data: {not json at all\n\n",
            b": a comment\n\n",
            b"\n",
            sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": f"ssn {SSN} end"}}),
        ]
    )
    assert_no_secrets(out)
    assert b"[REDACTED]" in out, "a malformed frame must not stop later frames being redacted"


async def test_a_non_json_frame_carrying_pii_is_still_redacted():
    """`data: [DONE]` is fine; `data: <raw PII>` must not be waved through."""
    out = await run([b"data: contact bob@secret.com\n\n"])
    assert_no_secrets(out)


async def test_the_done_sentinel_survives_intact():
    out = await run([b"data: [DONE]\n\n"])
    assert b"[DONE]" in out


@pytest.mark.parametrize("chunk", [1, 3, 17, 512])
async def test_no_transport_chunking_changes_the_answer(chunk):
    frames = [
        sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": f"ssn {SSN} and "}}),
        sse({"choices": [{"index": 0, "delta": {"content": f"mail {EMAIL} "}}]}),
        sse({"delta": {"type": "thinking_delta", "thinking": CARD}}),
        sse({"type": "content_block_stop"}, event="content_block_stop"),
    ]
    out = await run(frames, chunk=chunk)
    assert_no_secrets(out)


async def test_a_value_split_across_the_transport_boundary_is_redacted():
    """One byte per chunk, PII straddling both the SSE frame and the value."""
    frames = [
        sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "my ssn is 123-4"}}),
        sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "5-6789 done"}}),
    ]
    out = await run(frames, chunk=1)
    assert_no_secrets(out)
    assert b"[REDACTED]" in out


async def test_a_long_key_in_a_stream_never_reaches_the_client():
    """GR-1, end to end through the SSE layer rather than the redactor alone."""
    key = "sk-" + "A" * 900
    text = f"key {key} end"
    frames = [
        sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": text[i : i + 7]}})
        for i in range(0, len(text), 7)
    ]
    out = await run(frames, chunk=1)
    assert b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in out, "the key streamed out in the clear"
    assert b"[REDACTED]" in out


async def test_held_text_is_released_before_the_block_ends():
    """A response finishing mid-partial must not lose its tail."""
    frames = [
        sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "call me at 123-4"}}),
        sse({"type": "content_block_stop"}, event="content_block_stop"),
    ]
    out = await run(frames)
    assert b"123-4" in out, "a partial match is text, not PII, once the stream ends"


async def test_tool_arguments_and_prose_do_not_contaminate_each_other():
    """Separate buffers: interleaving two logical streams through one holdback
    would corrupt both."""
    frames = [
        sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "prefix 123-4"}}),
        sse({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a":"5-6789'}}),
        sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "5-6789 tail"}}),
        sse({"type": "content_block_stop"}, event="content_block_stop"),
    ]
    out = await run(frames)
    assert_no_secrets(out)
