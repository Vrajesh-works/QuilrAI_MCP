"""PII patterns and their validators.

Two properties matter for a streaming redactor and shape everything here:

* Every pattern is **left-anchored on a word boundary**. That makes the safe
  emit point always fall on a token boundary, so flushing text early can never
  manufacture a match that the full text would not have contained.
* Every pattern has a **bounded maximum length**. The holdback buffer is sized
  from it, which is what keeps memory constant regardless of stream length.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import regex

REDACTION_PLACEHOLDER = "[REDACTED]"


def luhn_valid(digits: str) -> bool:
    """The Luhn checksum every real card number satisfies.

    Without it, any 16-digit order id or tracking number is redacted as a credit
    card. That is a false positive users notice immediately and it erodes trust
    in the guardrail. The trade is explicit: a 16-digit string that fails Luhn is
    not a card number, so letting it through is not a PII leak.
    """
    numbers = [int(character) for character in digits if character.isdigit()]
    if len(numbers) < 13:
        return False
    checksum = 0
    parity = len(numbers) % 2
    for index, digit in enumerate(numbers):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


@dataclass(frozen=True)
class Pattern:
    """One detector.

    Attributes:
        max_length: the longest text this pattern can match. Sets the holdback
            bound, so it must be an over-estimate rather than an under-estimate.
        validate: optional second stage, applied to a full match to suppress
            false positives. Never used to *widen* a match.
    """

    name: str
    expression: regex.Pattern
    max_length: int
    validate: Callable[[str], bool] | None = None

    def accepts(self, text: str) -> bool:
        return self.validate is None or self.validate(text)


# Local part per RFC 5322 in practice, not in full generality: the exotic
# quoted-string forms are not worth the false positives they bring.
EMAIL = Pattern(
    name="email",
    expression=regex.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    max_length=320,  # RFC 5321: 64-char local part + @ + 255-char domain.
)

# Deliberately does not match 000/666/9xx area numbers, which are never issued.
SSN = Pattern(
    name="ssn",
    expression=regex.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    max_length=11,
)

# 13-19 digits, optionally grouped by a single space or hyphen. The separator is
# why a credit card cannot be found by "hold back the current word": a match
# spans spaces, so word boundaries are not safe cut points for this pattern.
CREDIT_CARD = Pattern(
    name="credit_card",
    expression=regex.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    max_length=25,
    validate=luhn_valid,
)

# North American numbers in the shapes a model actually emits. The word-boundary
# anchor sits inside each alternative rather than in front of the group: a
# leading `\b(` can never match, because `(` is not a word character, which
# silently disables the whole pattern for parenthesised area codes.
PHONE = Pattern(
    name="phone",
    expression=regex.compile(r"(?:\b\+1[ -]?)?(?:\(\d{3}\)[ .-]?|\b\d{3}[ .-]?)\d{3}[ .-]?\d{4}\b"),
    max_length=20,
)

# Provider-shaped API keys: leaking one of these in a response is as bad as PII.
API_KEY = Pattern(
    name="api_key",
    expression=regex.compile(r"\b(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
    max_length=128,
)

DEFAULT_PATTERNS: tuple[Pattern, ...] = (EMAIL, SSN, CREDIT_CARD, PHONE, API_KEY)

PATTERNS_BY_NAME = {pattern.name: pattern for pattern in DEFAULT_PATTERNS}
