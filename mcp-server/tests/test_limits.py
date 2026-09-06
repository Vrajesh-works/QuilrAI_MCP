"""Input limits and the Unicode reason policy, at their exact boundaries.

MCP-3: a 5 MB `reason` was accepted, emitted as a 10 MB response, and retained
forever. MCP-4: ten U+200B zero-width spaces satisfied the "meaningful reason"
rule on a money-moving tool.
"""

from __future__ import annotations

import json

import pytest
from customer_mcp.schemas import (
    MAX_IDEMPOTENCY_KEY_LENGTH,
    MAX_REASON_LENGTH,
    MIN_REASON_LENGTH,
    TriggerRefundInput,
    visible_length,
)
from pydantic import ValidationError

BASE = {"customer_id": "CUST-00042", "amount": 10.0}


def build(reason: str, **extra) -> TriggerRefundInput:
    return TriggerRefundInput(**BASE, reason=reason, **extra)


# --------------------------------------------------------------------------
# MCP-3 - upper bound on `reason`, at limit-1 / limit / limit+1 / very large
# --------------------------------------------------------------------------


def test_reason_at_one_below_the_limit_is_accepted():
    assert len(build("a" * (MAX_REASON_LENGTH - 1)).reason) == MAX_REASON_LENGTH - 1


def test_reason_exactly_at_the_limit_is_accepted():
    assert len(build("a" * MAX_REASON_LENGTH).reason) == MAX_REASON_LENGTH


def test_reason_one_over_the_limit_is_rejected():
    with pytest.raises(ValidationError):
        build("a" * (MAX_REASON_LENGTH + 1))


@pytest.mark.parametrize("size", [10_000, 1_000_000, 5_000_000])
def test_very_large_reasons_are_rejected_quickly(size):
    """The audit's exact three sizes. All three used to be accepted."""
    with pytest.raises(ValidationError):
        build("a" * size)


def test_an_oversized_reason_cannot_amplify_the_response():
    """The response was ~2x the input because the payload is serialised twice.
    Bounding the input is what bounds the amplification."""
    accepted = build("a" * MAX_REASON_LENGTH)
    serialised = json.dumps({"reason": accepted.reason}, indent=2)
    assert len(serialised) < 4 * MAX_REASON_LENGTH


def test_the_advertised_schema_states_the_limit():
    """`tools/list` must not promise something `tools/call` will refuse."""
    schema = TriggerRefundInput.model_json_schema()
    assert schema["properties"]["reason"]["maxLength"] == MAX_REASON_LENGTH


def test_the_idempotency_key_is_bounded():
    build("A perfectly good reason.", idempotency_key="k" * MAX_IDEMPOTENCY_KEY_LENGTH)
    with pytest.raises(ValidationError):
        build("A perfectly good reason.", idempotency_key="k" * (MAX_IDEMPOTENCY_KEY_LENGTH + 1))


# --------------------------------------------------------------------------
# MCP-4 - the "meaningful reason" rule over Unicode
# --------------------------------------------------------------------------

INVISIBLE = [
    pytest.param("​", id="U+200B zero-width space"),
    pytest.param("‌", id="U+200C zero-width non-joiner"),
    pytest.param("‍", id="U+200D zero-width joiner"),
    pytest.param("﻿", id="U+FEFF BOM"),
    pytest.param(" ", id="U+00A0 no-break space"),
    pytest.param("　", id="U+3000 ideographic space"),
    pytest.param(" ", id="U+2028 line separator"),
    pytest.param(" ", id="U+2029 paragraph separator"),
    pytest.param(" ", id="U+2007 figure space"),
    pytest.param("᠎", id="U+180E Mongolian vowel separator"),
    pytest.param(" ", id="ASCII space"),
    pytest.param("\t", id="tab"),
    pytest.param("\n", id="newline"),
    pytest.param("\r", id="carriage return"),
    pytest.param("\x0b", id="vertical tab"),
]


@pytest.mark.parametrize("character", INVISIBLE)
def test_a_reason_of_only_invisible_characters_is_rejected(character):
    with pytest.raises(ValidationError):
        build(character * 40)


def test_invisible_padding_cannot_pad_a_short_reason_up_to_the_minimum():
    with pytest.raises(ValidationError):
        build("short" + "​" * 200)


def test_mixed_invisible_and_visible_counts_only_the_visible():
    with pytest.raises(ValidationError):
        build("​".join("abcdefghi"))  # 9 visible characters
    accepted = build("​".join("abcdefghij"))  # 10 visible characters
    assert visible_length(accepted.reason) == MIN_REASON_LENGTH


LEGITIMATE = [
    pytest.param("Duplicate charge on the April invoice.", id="english"),
    pytest.param("Doppelte Belastung auf der Rechnung.", id="german"),
    pytest.param("Двойное списание по счёту клиента.", id="russian"),
    pytest.param("客户账单被重复扣款，需要退款。", id="chinese"),
    pytest.param("重複請求のため返金します。", id="japanese"),
    pytest.param("الرسوم مكررة على الفاتورة", id="arabic"),
    pytest.param("Χρέωση διπλή στο τιμολόγιο", id="greek"),
    pytest.param("Facturación duplicada señor", id="spanish-with-accents"),
    pytest.param("Refund 🙏 for the duplicate 💳 charge", id="emoji-mixed"),
    pytest.param("नकली शुल्क वापसी", id="devanagari"),
]


@pytest.mark.parametrize("reason", LEGITIMATE)
def test_legitimate_international_reasons_are_accepted(reason):
    """The rule must not quietly become 'must be English'."""
    assert build(reason).reason == reason.strip()


def test_combining_marks_count_as_visible():
    """`e` + U+0301 is a visible character, not decoration to be discounted."""
    decomposed = "é" * 6  # 6 visible letters, 12 code points
    assert visible_length(decomposed) == 12
    build(decomposed)


def test_a_reason_of_exactly_the_minimum_visible_length_is_accepted():
    assert len(build("a" * MIN_REASON_LENGTH).reason) == MIN_REASON_LENGTH


def test_one_below_the_minimum_is_rejected():
    with pytest.raises(ValidationError):
        build("a" * (MIN_REASON_LENGTH - 1))


def test_surrounding_whitespace_is_still_trimmed_from_what_is_stored():
    assert build("   1234567890   ").reason == "1234567890"


def test_an_invisible_only_idempotency_key_is_rejected_rather_than_ignored():
    """Silently treating it as 'no key' would turn a replay guard into none."""
    with pytest.raises(ValidationError):
        build("A perfectly good reason.", idempotency_key="​​")
    with pytest.raises(ValidationError):
        build("A perfectly good reason.", idempotency_key="   ")


def test_the_customer_id_bound_is_unchanged():
    """It was already right; this pins it against regression."""
    with pytest.raises(ValidationError):
        TriggerRefundInput(customer_id="CUST-" + "0" * 100_000, amount=1.0, reason="a" * 20)
