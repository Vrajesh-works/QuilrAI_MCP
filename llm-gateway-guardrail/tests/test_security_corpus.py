"""Hand-written security corpus. The implementation is never the oracle.

Why this file exists
--------------------
Every other correctness test in this suite asserts ``streamed == redact_text(text)``.
Both sides call the same pattern set, so the suite can only ever detect
*disagreement between two callers of the same patterns* - never a wrong pattern.
That is not a hypothetical weakness: it is exactly why the adjacency leak
(``bob@example.com123-45-6789``, redacted by neither path) survived 312 passing
tests. The oracle shared the defect, so the suite agreed with itself and both
sides leaked.

Here every expected output is **written by hand from the stated policy** and
compared against three independent things:

1. batch redaction == the hand-written expectation
2. streaming redaction == the hand-written expectation, at nine chunkings
3. streaming == batch (the split-invariance property the design claims)

plus a blunt fourth check that survives any future change to the placeholder or
the exact expected string: **no sensitive source substring appears in the
output**. `"[REDACTED]" in output` is not an assertion, it is a formality - a
leak of 500 characters with a placeholder appended would pass it.

Stated policy
-------------
* An email address, a US SSN, a Luhn-valid card number, a NANP phone number
  written with a disambiguating signal, and a provider-shaped API key are each
  replaced by ``[REDACTED]``.
* Two sensitive values written adjacently each get their own placeholder,
  **unless** the characters of one are legitimately part of the other's syntax -
  digits immediately before an ``@`` are an email local part, not a separate
  SSN, and one placeholder covering both is correct there.
* A candidate token run longer than the redactor's holdback ceiling is replaced
  by a single ``[REDACTED]`` on the streaming path. See `redactor.py`; the
  divergence from batch for *benign* over-long runs is deliberate and is pinned
  by `test_overlong_benign_run_fails_closed_while_streaming`.
"""

from __future__ import annotations

import random

import pytest
from llm_guardrail.patterns import DEFAULT_PATTERNS, OVERLONG_RUN, UNBOUNDED_PATTERNS
from llm_guardrail.redactor import StreamRedactor, redact_text

R = "[REDACTED]"


# --------------------------------------------------------------------------
# Chunking strategies. The point is that *none* of them may change the answer.
# --------------------------------------------------------------------------


def _chunkings(text: str) -> dict[str, list[str]]:
    """Every split shape the brief calls for, named so failures are readable."""
    n = len(text)
    shapes: dict[str, list[str]] = {
        "whole": [text],
        "one-character": list(text),
        "two": [text[: n // 2], text[n // 2 :]],
        "three": [text[: n // 3], text[n // 3 : 2 * n // 3], text[2 * n // 3 :]],
        "size-4": [text[i : i + 4] for i in range(0, n, 4)],
        "size-40": [text[i : i + 40] for i in range(0, n, 40)],
    }

    # Deterministic "random" chunking: seeded so a failure is reproducible.
    rng = random.Random(20260906)
    pieces, position = [], 0
    while position < n:
        step = rng.randint(1, 7)
        pieces.append(text[position : position + step])
        position += step
    shapes["random"] = pieces

    # Every single split point. This is the exhaustive version and it subsumes
    # "split inside the prefix", "split inside the token" and "split immediately
    # before the sensitive suffix" - all of them are some value of i.
    for i in range(1, n):
        shapes[f"split@{i}"] = [text[:i], text[i:]]
    return shapes


def _stream(text: str, chunks: list[str]) -> str:
    return StreamRedactor().process(chunks)


def _assert_corpus_entry(text: str, expected: str, forbidden: tuple[str, ...] = ()) -> None:
    """The four independent checks described in the module docstring."""
    batch = redact_text(text)
    assert batch == expected, f"batch redaction of {text!r}\n  got      {batch!r}\n  expected {expected!r}"

    for name, chunks in _chunkings(text).items():
        assert "".join(chunks) == text, f"chunking {name} is not a partition of the input"
        streamed = _stream(text, chunks)
        assert streamed == expected, (
            f"streaming {text!r} as {name}\n  got      {streamed!r}\n  expected {expected!r}"
        )
        assert streamed == batch, f"streaming/batch divergence for {text!r} at {name}"
        for secret in forbidden:
            assert secret not in streamed, f"{secret!r} survived {name} chunking of {text!r}"
            assert secret not in batch, f"{secret!r} survived batch redaction of {text!r}"


# --------------------------------------------------------------------------
# Baseline: the values the brief names, in ordinary prose.
# --------------------------------------------------------------------------

BASELINE: list[tuple[str, str, tuple[str, ...]]] = [
    (
        "Contact john.doe@example.com for help.",
        f"Contact {R} for help.",
        ("john.doe@example.com", "john.doe"),
    ),
    ("my ssn is 123-45-6789 ok", f"my ssn is {R} ok", ("123-45-6789",)),
    ("card 4111 1111 1111 1111 end", f"card {R} end", ("4111 1111 1111 1111",)),
    ("card 4111-1111-1111-1111 end", f"card {R} end", ("4111-1111-1111-1111",)),
    ("card 4111111111111111 end", f"card {R} end", ("4111111111111111",)),
    ("call (555) 123-4567 now", f"call {R} now", ("(555) 123-4567",)),
    ("call 555-123-4567 now", f"call {R} now", ("555-123-4567",)),
    (
        "key sk-ant-abcdefghij1234567890 end",
        f"key {R} end",
        ("sk-ant-abcdefghij1234567890",),
    ),
    ("key AKIAIOSFODNN7EXAMPLE end", f"key {R} end", ("AKIAIOSFODNN7EXAMPLE",)),
    # The brief's own worked example.
    ("Contact john.doe@exa" + "mple.com for help.", f"Contact {R} for help.", ("john.doe@example.com",)),
]


@pytest.mark.parametrize(("text", "expected", "forbidden"), BASELINE, ids=lambda v: None)
def test_baseline_values_are_redacted_identically_at_every_chunking(text, expected, forbidden):
    _assert_corpus_entry(text, expected, forbidden)


# --------------------------------------------------------------------------
# GR-2 - adjacency. The whole matrix the brief asks for.
# --------------------------------------------------------------------------

EMAIL = "bob@example.com"
SSN = "123-45-6789"
SSN2 = "078-05-1120"
CARD = "4111111111111111"

ADJACENCY: list[tuple[str, str, tuple[str, ...]]] = [
    # --- no separator at all: the case that defeated `\b` on both paths.
    (f"{EMAIL}{SSN}", f"{R}{R}", (EMAIL, SSN)),
    ("a@b.com123-45-6789", f"{R}{R}", ("a@b.com", SSN)),
    (f"{EMAIL}{CARD}", f"{R}{R}", (EMAIL, CARD)),
    (f"{SSN}{CARD}", f"{R}{R}", (SSN, CARD)),
    (f"{CARD}{SSN}", f"{R}{R}", (SSN, CARD)),
    (f"{SSN}{SSN2}", f"{R}{R}", (SSN, SSN2)),
    (f"Contact {EMAIL}{SSN} now", f"Contact {R}{R} now", (EMAIL, SSN)),
    # PII touching an ordinary letter on either side.
    (f"x{SSN}", f"x{R}", (SSN,)),
    (f"{SSN}x", f"{R}x", (SSN,)),
    (f"ref{SSN}end", f"ref{R}end", (SSN,)),
    # --- ordinary separators.
    (f"{EMAIL} {SSN}", f"{R} {R}", (EMAIL, SSN)),
    (f"{SSN} {EMAIL}", f"{R} {R}", (EMAIL, SSN)),
    (f"{EMAIL}, {SSN}", f"{R}, {R}", (EMAIL, SSN)),
    (f"{EMAIL};{SSN}", f"{R};{R}", (EMAIL, SSN)),
    (f"{EMAIL}\n{SSN}", f"{R}\n{R}", (EMAIL, SSN)),
    (f"{EMAIL}\t{SSN}", f"{R}\t{R}", (EMAIL, SSN)),
    (f"{EMAIL}|{SSN}", f"{R}|{R}", (EMAIL, SSN)),
    (f"{CARD} {SSN}", f"{R} {R}", (CARD, SSN)),
    (f"{SSN} {CARD}", f"{R} {R}", (CARD, SSN)),
    (f"{EMAIL} {EMAIL}", f"{R} {R}", (EMAIL,)),
    (f"{SSN}, {SSN2}", f"{R}, {R}", (SSN, SSN2)),
    # --- Unicode punctuation between two values.
    (f"{EMAIL}—{SSN}", f"{R}—{R}", (EMAIL, SSN)),  # em dash
    (f"{EMAIL}…{SSN}", f"{R}…{R}", (EMAIL, SSN)),  # ellipsis
    (f"{EMAIL} {SSN}", f"{R} {R}", (EMAIL, SSN)),  # line separator
    (f"{EMAIL}、{SSN}", f"{R}、{R}", (EMAIL, SSN)),  # ideographic comma
    # --- inside structured text, which is how this happens by accident.
    (f'{{"email":"{EMAIL}","ssn":"{SSN}"}}', f'{{"email":"{R}","ssn":"{R}"}}', (EMAIL, SSN)),
    (f"{EMAIL},{SSN},{CARD}", f"{R},{R},{R}", (EMAIL, SSN, CARD)),
]


@pytest.mark.parametrize(("text", "expected", "forbidden"), ADJACENCY, ids=lambda v: None)
def test_adjacent_pii_is_redacted_on_both_paths(text, expected, forbidden):
    _assert_corpus_entry(text, expected, forbidden)


def test_an_ssn_flush_against_an_address_yields_two_placeholders():
    """`123-45-6789bob@example.com` has two defensible readings.

    It is one email address whose local part happens to contain an SSN, and it
    is also an SSN written flush against an address. The redactor takes the
    second reading: the address cannot be settled while it still touches the end
    of the buffer and might grow, so the SSN inside it resolves first and the
    address resolves afterwards.

    Two placeholders is the reading this corpus specifies, because it is the one
    that stays correct if the input is truncated - and because for a detector,
    resolving an ambiguity into *more* redaction is the direction that cannot
    cause a breach. What is asserted below regardless of reading is that neither
    value survives and that every chunking agrees.
    """
    text = "123-45-6789bob@example.com"
    _assert_corpus_entry(text, f"{R}{R}", (SSN, "bob@example.com", "example.com"))


def test_two_addresses_with_no_separator_lose_both_local_parts():
    """A documented imperfection, pinned so it cannot silently get worse.

    `bob@example.comalice@example.org` has no unambiguous parse - `comalice`
    is a syntactically valid TLD. The redactor consumes `bob@example.comalice`,
    so both local parts and the first domain are destroyed and neither complete
    address survives. The trailing `@example.org` is a bare domain, not an
    address, and is not treated as PII by any pattern here.
    """
    text = "bob@example.comalice@example.org"
    batch = redact_text(text)
    for secret in ("bob@example.com", "alice@example.org", "alice", "bob"):
        assert secret not in batch
    for name, chunks in _chunkings(text).items():
        assert _stream(text, chunks) == batch, f"divergence at {name}"


# --------------------------------------------------------------------------
# GR-1 - long tokens. The core of this remediation.
# --------------------------------------------------------------------------

LONG_TOKEN_LENGTHS = (100, 300, 320, 327, 328, 329, 500, 1000, 5000, 10000)


@pytest.mark.parametrize("length", LONG_TOKEN_LENGTHS)
def test_a_long_api_key_never_reaches_the_client(length):
    """The GR-1 regression. Below 327 this always worked; at and above it the
    key streamed out in full with no placeholder at all."""
    key = "sk-" + "A" * length
    text = f"here is a key {key} end"
    _assert_corpus_entry(text, f"here is a key {R} end", (key, "A" * 60, "sk-A"))


@pytest.mark.parametrize("length", LONG_TOKEN_LENGTHS)
def test_a_long_email_local_part_never_reaches_the_client(length):
    local = "a" * length
    text = f"mail {local}@example.com end"
    _assert_corpus_entry(text, f"mail {R} end", (f"{local}@example.com", "a" * 60))


@pytest.mark.parametrize("length", (400, 1000, 5000))
def test_a_long_anthropic_key_never_reaches_the_client(length):
    key = "sk-ant-" + "B" * length
    text = f"key {key} stop"
    _assert_corpus_entry(text, f"key {R} stop", (key, "B" * 60))


def test_the_holdback_stays_bounded_while_a_long_token_streams():
    """Fail-closed must not be bought with unbounded memory."""
    redactor = StreamRedactor()
    ceiling = redactor._max_holdback
    for character in "here is a key sk-" + "A" * 20000 + " end":
        redactor.feed(character)
        assert len(redactor.pending) <= ceiling, "buffer grew past the ceiling"
    redactor.flush()
    assert redactor.stats.max_holdback_seen <= ceiling
    assert redactor.stats.counts.get("overflow", 0) >= 1, "the fail-closed path should have fired"


def test_overlong_benign_run_fails_closed_while_streaming():
    """The deliberate trade, stated as a test so nobody rediscovers it live.

    A 500-character run of token characters that is *not* PII is left alone by
    the batch path and replaced by a placeholder while streaming. Streaming
    cannot know the run will never become an address, and guessing in the
    direction of release is what GR-1 was.
    """
    blob = "Q" * 500
    text = f"data {blob} end"
    assert redact_text(text) == text, "batch should leave a benign run alone"
    streamed = _stream(text, list(text))
    assert streamed == f"data {R} end"
    assert blob[:60] not in streamed


def test_a_long_run_is_suppressed_rather_than_released_after_the_placeholder():
    """A placeholder followed by the rest of the secret is not a fix."""
    key = "sk-" + "Z" * 900
    redactor = StreamRedactor()
    out = "".join([*(redactor.feed(c) for c in f"key {key} end"), redactor.flush()])
    assert out == f"key {R} end"
    assert "Z" not in out
    assert redactor.stats.suppressed_characters > 0


# --------------------------------------------------------------------------
# GR-4 - international addresses.
# --------------------------------------------------------------------------

UNICODE_EMAILS = [
    ("mail josé.garcía@example.com now", "josé.garcía@example.com"),
    ("to françois.müller@example.de today", "françois.müller@example.de"),
    ("владимир@почта.рф", None),
    ("write user@exämple.com back", "user@exämple.com"),
]


@pytest.mark.parametrize(("text", "address"), UNICODE_EMAILS, ids=lambda v: None)
def test_international_addresses_are_redacted_at_every_chunking(text, address):
    expected = text.replace(address, R) if address else R
    _assert_corpus_entry(text, expected, (address,) if address else ())


# --------------------------------------------------------------------------
# False positives. Under-redaction is a breach; over-redaction is a bug report.
# --------------------------------------------------------------------------

BENIGN = [
    "Your order number is 1234567890 and ships today.",
    "invoice 5551234567 was paid",
    "version 1.2.3 build 4567890123",
    "Your order number is 1234567890123456 and it shipped.",  # 16 digits, fails Luhn
    "customer CUST-00042 has 3 open tickets",
    "the answer is 42",
    "sha 3b2c1d4e5f60718293a4b5c6d7e8f9a0",  # 32 hex chars, under the ceiling
    "read RFC 5322 for the grammar",
    "call 2024-05-12 the cutoff date",
]


@pytest.mark.parametrize("text", BENIGN, ids=lambda v: None)
def test_benign_text_passes_through_untouched(text):
    _assert_corpus_entry(text, text)


def test_a_luhn_valid_thirteen_digit_number_is_treated_as_a_card():
    """An unavoidable collision, written down rather than left to surprise.

    `1738368000000` is an epoch-milliseconds timestamp. It is also thirteen
    digits that satisfy Luhn, which is exactly the shape of a thirteen-digit
    Visa. No detector can separate the two, so this one is redacted. Luhn still
    does its job on the far more common sixteen-digit case, where a random order
    number has a one-in-ten chance of colliding rather than a certainty.
    """
    from llm_guardrail.patterns import luhn_valid

    assert luhn_valid("1738368000000"), "premise of this test"
    _assert_corpus_entry("epoch 1738368000000 in millis", f"epoch {R} in millis")


def test_a_non_luhn_timestamp_survives():
    from llm_guardrail.patterns import luhn_valid

    assert not luhn_valid("1738368000001"), "premise of this test"
    _assert_corpus_entry("epoch 1738368000001 in millis", "epoch 1738368000001 in millis")


def test_a_card_written_next_to_an_ssn_is_still_found():
    """The greedy-match-fails-Luhn regression.

    The card pattern swallows the separator and the SSN's first three digits,
    producing a nineteen-digit candidate that fails Luhn. Discarding that
    candidate outright - which is what the redactor used to do - left a valid
    sixteen-digit card sitting in the clear beside a redacted SSN.
    """
    _assert_corpus_entry(f"{CARD} {SSN}", f"{R} {R}", (CARD, SSN))
    _assert_corpus_entry(f"{CARD} 123", f"{R} 123", (CARD,))
    _assert_corpus_entry(f"{CARD} 4444", f"{R} 4444", (CARD,))


# --------------------------------------------------------------------------
# The assumptions the redactor's fail-closed path is built on, asserted rather
# than trusted. If either of these breaks, GR-1's fix is unsound.
# --------------------------------------------------------------------------


def test_bounded_patterns_cannot_outgrow_their_declared_max_length():
    """SSN, card and phone must be bounded by construction, because the
    suppression logic assumes only EMAIL and API_KEY can overflow."""
    bounded = [p for p in DEFAULT_PATTERNS if p not in UNBOUNDED_PATTERNS]
    assert {p.name for p in bounded} == {"ssn", "credit_card", "phone"}
    haystacks = [
        "1" * 4000,
        "1-" * 2000,
        "1 " * 2000,
        "(123) " + "4" * 2000,
        "+1-" + "5-" * 2000,
        ("123-45-6789" * 300),
    ]
    for pattern in bounded:
        for haystack in haystacks:
            for match in pattern.expression.finditer(haystack):
                assert len(match.group()) <= pattern.max_length, (
                    f"{pattern.name} matched {len(match.group())} chars, "
                    f"declared max is {pattern.max_length}"
                )


def test_the_validator_floor_cannot_reject_something_the_validator_accepts():
    """`validate_min_length` is a performance floor on the shorter-candidate
    retry. Set too high it would silently lose detections, so it is checked
    against the validator itself rather than trusted."""
    from llm_guardrail.patterns import luhn_valid

    for pattern in DEFAULT_PATTERNS:
        if pattern.validate is None:
            assert pattern.validate_min_length == 0, pattern.name
            continue
        assert pattern.validate is luhn_valid, "a new validator needs its own floor check"
        # Nothing shorter than the floor may pass the validator.
        for length in range(1, pattern.validate_min_length):
            assert not luhn_valid("4" * length), f"luhn accepted {length} characters"
        # And the floor is not set above a real minimum: a 13-digit card exists.
        assert luhn_valid("4222222222222"), "13-digit Visa must still validate"
        assert len("4222222222222") == pattern.validate_min_length


def test_overlong_runs_are_confined_to_the_continuation_class():
    """Suppression ends a run at the first character outside `OVERLONG_RUN`.
    That is only safe if an over-long match contains nothing else."""
    samples = [
        "sk-" + "A" * 900,
        "sk-ant-" + "b_c-" * 300,
        "a" * 900 + "@example.com",
        "first.last+tag%2f" * 60 + "@sub.domain.example.co.uk",
    ]
    for sample in samples:
        matched = OVERLONG_RUN.match(sample)
        assert matched is not None and matched.end() == len(sample), (
            f"{sample[:40]!r}... contains a character outside the continuation class"
        )
    for pattern in UNBOUNDED_PATTERNS:
        for sample in samples:
            for match in pattern.expression.finditer(sample):
                run = OVERLONG_RUN.match(match.group())
                assert run is not None and run.end() == len(match.group())
