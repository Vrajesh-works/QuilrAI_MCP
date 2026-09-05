"""The core algorithm: redaction must not depend on where chunks fall.

The oracle throughout is whole-text redaction. If feeding a string one character
at a time gives a different answer than redacting it in one go, the streaming
implementation is broken - and that difference is exactly the bug the naive
per-chunk approach has.
"""

from __future__ import annotations

import random

import pytest

from llm_guardrail.redactor import StreamRedactor, redact_text

SECRETS = {
    "email": "ada.lovelace@example.com",
    "ssn": "123-45-6789",
    "credit_card": "4111 1111 1111 1111",
    "phone": "(555) 123-4567",
    "api_key": "sk-ant-abcdefghij1234567890",
}


def chunk_every(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)] or [""]


@pytest.mark.parametrize("name,secret", SECRETS.items())
def test_each_pattern_is_redacted_in_one_piece(name, secret):
    output = redact_text(f"the value is {secret} okay")
    assert secret not in output
    assert "[REDACTED]" in output


@pytest.mark.parametrize("name,secret", SECRETS.items())
@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11, 64])
def test_redaction_survives_every_chunk_size(name, secret, size):
    """The core property: a match split across chunks must still be caught."""
    text = f"Please contact using {secret} before Friday."
    streamed = StreamRedactor().process(chunk_every(text, size))

    assert secret not in streamed, f"{name} leaked when chunked every {size} chars"
    assert streamed == redact_text(text), "streamed output diverged from whole-text redaction"


@pytest.mark.parametrize("name,secret", SECRETS.items())
def test_split_at_every_single_offset(name, secret):
    """Exhaustive: break the text at each position in turn.

    A boundary bug usually hides at one specific offset - the middle of an SSN,
    just after the '@' - so every split point is checked rather than sampled.
    """
    text = f"value {secret} end"
    expected = redact_text(text)

    for offset in range(1, len(text)):
        streamed = StreamRedactor().process([text[:offset], text[offset:]])
        assert streamed == expected, f"{name} broke when split at offset {offset}: {streamed!r}"


def test_randomised_chunkings_agree_with_whole_text_redaction():
    text = (
        "Reach Ada at ada@example.com or 555-123-4567. Card 4111-1111-1111-1111, "
        "SSN 123-45-6789, key sk-ant-abcdefghij1234567890. Order 1234567890123456 is fine."
    )
    expected = redact_text(text)
    rng = random.Random(20260905)

    for _ in range(200):
        chunks, position = [], 0
        while position < len(text):
            step = rng.randint(1, 9)
            chunks.append(text[position : position + step])
            position += step
        assert StreamRedactor().process(chunks) == expected


def test_multiple_secrets_in_one_stream():
    text = f"{SECRETS['email']} and {SECRETS['ssn']} and {SECRETS['credit_card']}"
    redactor = StreamRedactor()
    output = redactor.process(chunk_every(text, 3))

    assert output.count("[REDACTED]") == 3
    assert redactor.stats.total == 3
    assert set(redactor.stats.counts) == {"email", "ssn", "credit_card"}


def test_clean_text_passes_through_byte_identical():
    text = "The quick brown fox jumps over the lazy dog. Nothing sensitive here at all."
    assert StreamRedactor().process(chunk_every(text, 4)) == text


def test_secret_at_the_very_end_is_flushed():
    """A response ending mid-partial-match must not silently lose its tail."""
    redactor = StreamRedactor()
    output = redactor.process(["my ssn is ", "123-45-", "6789"])

    assert output == "my ssn is [REDACTED]"
    assert redactor.pending == "", "flush must leave nothing held"


def test_partial_that_never_completes_is_released_verbatim():
    """Text that merely looked like PII must come out unchanged, not dropped."""
    output = StreamRedactor().process(["the number 123-45-", " was wrong"])
    assert output == "the number 123-45- was wrong"


def test_forgetting_to_flush_is_the_only_way_to_lose_text():
    """Documents the contract: feed() alone may withhold the tail."""
    redactor = StreamRedactor()
    emitted = redactor.feed("ends with 123-45")

    assert "123-45" not in emitted
    assert redactor.pending == "123-45"
    assert redactor.flush() == "123-45"


class TestBoundedMemory:
    """Memory must not grow with stream length - the second scored property."""

    def test_holdback_stays_bounded_over_a_long_stream(self):
        redactor = StreamRedactor()
        for _ in range(2_000):
            redactor.feed("some ordinary prose that contains nothing sensitive at all. ")

        assert len(redactor.pending) < 64
        assert redactor.stats.max_holdback_seen <= redactor._max_holdback

    def test_adversarial_digit_stream_cannot_grow_the_buffer(self):
        """A long run of digits looks like a forever-growing partial card number.

        Without the ceiling this is an unbounded buffer - a memory exhaustion
        vector triggerable by asking the model to count.
        """
        redactor = StreamRedactor()
        for _ in range(500):
            redactor.feed("1234567890")

        assert len(redactor.pending) <= redactor._max_holdback
        assert redactor.stats.max_holdback_seen <= redactor._max_holdback

    def test_output_length_tracks_input_for_clean_text(self):
        redactor = StreamRedactor()
        text = "nothing to see here, move along. " * 100
        output = redactor.process(chunk_every(text, 16))

        assert output == text
        assert redactor.stats.characters_out == redactor.stats.characters_in


class TestFalsePositives:
    """Over-redaction is a real cost; these pin the deliberate trade-offs."""

    def test_non_luhn_sixteen_digit_number_is_left_alone(self):
        text = "Your order number is 1234567890123456 and it shipped."
        assert redact_text(text) == text

    def test_luhn_valid_card_is_still_redacted(self):
        assert "4111" not in redact_text("card 4111 1111 1111 1111 here")

    @pytest.mark.parametrize("invalid", ["000-12-3456", "666-12-3456", "900-12-3456", "123-00-4567", "123-45-0000"])
    def test_never_issued_ssn_ranges_are_not_matched(self, invalid):
        text = f"the code {invalid} is not an ssn"
        assert redact_text(text) == text

    def test_ordinary_prose_with_numbers_survives(self):
        text = "In 2026 we processed 45 orders across 3 regions, up 12 percent."
        assert redact_text(text) == text

    def test_version_strings_are_not_phone_numbers(self):
        text = "Upgrade to 1.2.3 or later."
        assert redact_text(text) == text
