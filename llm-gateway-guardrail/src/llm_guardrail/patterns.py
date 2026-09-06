"""PII patterns and their validators.

Anchoring policy
----------------
No pattern here is wrapped in ``\\b`` at both ends. ``\\b`` is a *mutual*
assertion between two adjacent characters, which is the wrong tool for a PII
detector: writing two sensitive values back to back, or letting one touch an
ordinary letter, destroys the boundary and **both** matches are lost.

    'bob@example.com123-45-6789'  ->  nothing matches at all

That shape needs no attacker - a model emitting a table, a CSV row or a JSON
string produces it by accident.

Every anchor here is instead a **one-sided, non-consuming lookaround** chosen
per pattern, so that a match ending flush against the next value does not
prevent that next value from matching in its own right. Where a guard would
cost detections it is dropped entirely and the pattern is allowed to
over-match: a false redaction is visible and annoying, a missed one is a silent
breach.

Length policy
-------------
``max_length`` sizes the redactor's holdback buffer. The quantifiers here are
deliberately **not** bounded to it. A bounded quantifier would make a long
credential unmatchable rather than matchable-and-held, which enlarges a
streaming leak rather than containing it. The redactor handles runs longer than
the ceiling by failing closed instead; see ``redactor.py``.
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
        validate_min_length: shortest text `validate` could ever accept. When a
            greedy match fails the validator the redactor retries shorter
            candidates at the same start; this is where that search stops.
            Purely a performance floor - setting it too low costs time, and
            setting it above the true minimum would lose detections, so it is
            asserted against the validator in the tests.
    """

    name: str
    expression: regex.Pattern
    max_length: int
    validate: Callable[[str], bool] | None = None
    validate_min_length: int = 0

    def accepts(self, text: str) -> bool:
        return self.validate is None or self.validate(text)


# Local part per RFC 5322 in practice, not in full generality: the exotic
# quoted-string forms are not worth the false positives they bring.
#
# `\w` rather than `[A-Za-z0-9]` because `regex` makes `\w` Unicode-aware. An
# ASCII-only local part would leave `josé.garcía@example.com` unredacted while
# still matching when a chunk boundary landed inside it, so the same input
# would give different answers depending on where the network split it.
#
# The TLD is `[^\W\d_]{2,}` - Unicode letters, explicitly no digits. That is
# load-bearing for adjacency: in `bob@example.com123-45-6789` the domain run
# `example.com123` backtracks to `.com`, so the match ends at the real TLD and
# leaves `123-45-6789` free to be matched by SSN on the next pass.
#
# No trailing anchor: whatever follows the TLD is someone else's problem, and
# forbidding a word character there is precisely what lost both matches before.
EMAIL = Pattern(
    name="email",
    expression=regex.compile(r"(?<![\w.%+-])[\w.%+-]+@[\w.-]+\.[^\W\d_]{2,}"),
    max_length=320,  # RFC 5321: 64-char local part + @ + 255-char domain.
)

# Deliberately does not match 000/666/9xx area numbers, which are never issued.
#
# No lookarounds at all, on purpose: `(?<!\d)` would lose the second of two
# concatenated SSNs (`123-45-6789078-05-1120`) and `(?!\d)` would lose the
# first. The cost is over-matching inside a longer digit-and-hyphen run; the
# benefit is that no arrangement of adjacent values hides one. For a detector
# whose failure mode is a silent breach that is the right trade, and the shapes
# that would collide - `4111-1111-1111-1111`, `555-123-4567`, `2024-05-12` -
# cannot produce a `ddd-dd-dddd` match anyway.
SSN = Pattern(
    name="ssn",
    expression=regex.compile(r"(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}"),
    max_length=11,
)

# 13-19 digits, optionally grouped by a single space or hyphen. The separator is
# why a credit card cannot be found by "hold back the current word": a match
# spans spaces, so word boundaries are not safe cut points for this pattern.
#
# Digit guards on both sides rather than `\b`, so a card written flush against
# a letter still matches while a 20-digit order number still does not.
CREDIT_CARD = Pattern(
    name="credit_card",
    expression=regex.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    max_length=37,  # 19 digits + 18 separators.
    validate=luhn_valid,
    # `luhn_valid` returns False below 13 digits, so nothing shorter than 13
    # characters can ever be accepted.
    validate_min_length=13,
)

# North American numbers in the shapes a model actually emits. The anchor sits
# inside each alternative rather than in front of the group: a leading `\b(`
# can never match, because `(` is not a word character, which would disable the
# whole pattern for parenthesised area codes.
#
# Phone keeps a full non-word guard on the left, unlike SSN. It is the one
# pattern with no validator and the one most likely to corrupt benign text, so
# it is the one place where widening the anchor would cost more than it buys.
# Phone is also not a stated requirement - it is included voluntarily.
#
# It requires a **disambiguating signal**: a `+1`, a parenthesised area code, or
# a separator between the area code and the exchange. A bare ten-digit run does
# not match. `CREDIT_CARD` has `luhn_valid` for the same reason: without a
# second stage, order ids, invoice numbers, epoch milliseconds and build
# numbers are replaced by `[REDACTED]`, which is the failure users notice and
# the one that creates pressure to turn the guardrail off entirely. The cost is
# a real number written as ten bare digits going unredacted; that shape is
# indistinguishable from an order number, so no detector could separate them.
PHONE = Pattern(
    name="phone",
    expression=regex.compile(
        r"(?<![\w-])(?:"
        r"\+1[ .-]?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}"  # +1-prefixed, any grouping
        r"|\(\d{3}\)[ .-]?\d{3}[ .-]?\d{4}"  # parenthesised area code
        r"|\d{3}[ .-]\d{3}[ .-]?\d{4}"  # separated groups
        r")(?!\d)"
    ),
    max_length=20,
)

# Provider-shaped API keys: leaking one of these in a response is as bad as PII.
API_KEY = Pattern(
    name="api_key",
    expression=regex.compile(
        r"(?<![A-Za-z0-9])(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})(?![A-Za-z0-9])"
    ),
    max_length=128,
)

DEFAULT_PATTERNS: tuple[Pattern, ...] = (EMAIL, SSN, CREDIT_CARD, PHONE, API_KEY)

PATTERNS_BY_NAME = {pattern.name: pattern for pattern in DEFAULT_PATTERNS}

# Characters that can appear inside a match longer than the holdback ceiling.
#
# Only EMAIL and API_KEY have quantifiers that can produce a match longer than
# the ceiling - SSN, CREDIT_CARD and PHONE are bounded by construction at 11, 37
# and 20 characters. Both of those two draw exclusively from this class and
# contain no whitespace, so an over-long candidate is necessarily one unbroken
# run of these characters. `redactor.py` relies on that to know where such a run
# ends after it has decided to suppress it; the test suite asserts the property
# rather than trusting this comment.
OVERLONG_RUN = regex.compile(r"[\w.%+@-]+")

# The patterns that can outgrow the ceiling, named so that assertion can be
# written without restating the reasoning.
UNBOUNDED_PATTERNS: tuple[Pattern, ...] = (EMAIL, API_KEY)
