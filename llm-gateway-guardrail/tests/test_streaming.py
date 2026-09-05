"""End-to-end: PII must not reach the client, whatever the provider's chunking."""

from __future__ import annotations

import json

import pytest

from conftest import collect_raw, collect_text, request_body
from llm_guardrail.redactor import StreamRedactor, redact_text
from llm_guardrail.sse import SSEParser
from llm_guardrail.stream import redact_sse_stream

pytestmark = pytest.mark.asyncio

SECRETS = [
    "ada.lovelace@example.com",
    "123-45-6789",
    "4111 1111 1111 1111",
    "(555) 123-4567",
    "sk-ant-abcdefghij1234567890",
]


@pytest.mark.parametrize("secret", SECRETS)
@pytest.mark.parametrize("chunk_size", [1, 3, 8, 64])
async def test_secrets_never_reach_the_client(guardrail, secret, chunk_size):
    text = f"Here is the detail you asked for: {secret}. Let me know if that helps."

    response = await guardrail.post("/v1/messages", json=request_body(text, chunk_size=chunk_size))
    received = await collect_text(response)

    assert secret not in received
    assert "[REDACTED]" in received
    assert received == redact_text(text), "streamed result must match whole-text redaction"


async def test_raw_bytes_on_the_wire_contain_no_pii(guardrail):
    """Assert on the wire itself, not just the reassembled text.

    A leak could hide in an event the reassembler ignores, so this checks every
    byte the client actually receives.
    """
    text = "SSN 123-45-6789 and email ada@example.com and card 4111 1111 1111 1111."
    response = await guardrail.post("/v1/messages", json=request_body(text, chunk_size=2))

    raw = await collect_raw(response)

    for secret in ("123-45-6789", "ada@example.com", "4111"):
        assert secret.encode() not in raw, f"{secret} appeared on the wire"


async def test_clean_response_is_passed_through_unchanged(guardrail):
    text = "The capital of France is Paris. It has about two million residents."
    response = await guardrail.post("/v1/messages", json=request_body(text, chunk_size=5))

    assert await collect_text(response) == text


async def test_protocol_events_are_preserved(guardrail):
    """The guardrail rewrites text, not the protocol around it."""
    response = await guardrail.post("/v1/messages", json=request_body("Hello there, friend."))

    parser = SSEParser()
    events = []
    async for chunk in response.aiter_bytes():
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())

    names = [event.event for event in events]
    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_stop" in names
    assert names[-1] == "message_stop"


async def test_content_type_is_still_event_stream(guardrail):
    response = await guardrail.post("/v1/messages", json=request_body("hi"))
    assert response.headers["content-type"].startswith("text/event-stream")


async def test_secret_at_the_very_end_of_the_stream_is_redacted(guardrail):
    """The held tail must be flushed before content_block_stop, not stranded."""
    text = "The account number you wanted is 4111 1111 1111 1111"
    response = await guardrail.post("/v1/messages", json=request_body(text, chunk_size=3))
    received = await collect_text(response)

    assert "4111" not in received
    assert received.endswith("[REDACTED]")


async def test_non_streaming_response_is_also_redacted(guardrail):
    """`stream: false` returns one JSON body; it needs redacting too."""
    text = "Reach me at ada@example.com or 123-45-6789."
    response = await guardrail.post("/v1/messages", json=request_body(text, stream=False))

    body = response.json()
    assert body["content"][0]["text"] == redact_text(text)
    assert "ada@example.com" not in json.dumps(body)


async def test_thinking_deltas_are_not_treated_as_response_text():
    """Only text_delta carries response text; rewriting others corrupts a block."""

    async def upstream():
        payload = {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "x"}}
        yield f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()

    out = b""
    async for chunk in redact_sse_stream(upstream(), StreamRedactor()):
        out += chunk

    assert b"thinking_delta" in out
    assert b"thinking" in out


async def test_upstream_failure_is_sanitised(unreachable_guardrail):
    response = await unreachable_guardrail.post("/v1/messages", json=request_body("hi"))

    assert response.status_code == 502
    serialized = json.dumps(response.json())
    assert "10.4.5.6" not in serialized
    assert "internal-llm" not in serialized
    assert "Traceback" not in serialized


async def test_health_endpoint(guardrail):
    response = await guardrail.get("/healthz")
    assert response.json()["status"] == "ok"
