"""Provider definitions and the single-attempt call.

One attempt against one provider, with its own timeout, classified into an
outcome the router can act on. Deciding *what to do next* lives in `router.py`;
this module only reports what happened.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from llm_router.errors import log_and_sanitise

logger = logging.getLogger(__name__)


class Outcome(enum.StrEnum):
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"  # upstream 429
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"  # connection failure or 5xx
    CLIENT_ERROR = "client_error"  # 4xx that is the caller's fault


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    model: str
    # Per-provider, so a fallback can be given a tighter or looser budget than
    # the primary. 3s is long enough for a healthy provider to start responding
    # and short enough that a stuck one does not hold the client.
    timeout_seconds: float = 3.0
    # The gateway's own upstream credential. `app.py` strips the caller's key
    # and this supplies the replacement; both halves are required, since
    # stripping alone would 401 against any real endpoint and only appears to
    # work because both providers in the demo are unauthenticated mocks.
    #
    # Resolved from the environment at startup, never from a request header, so
    # a caller cannot influence which credential is used upstream.
    api_key: str | None = None
    #: Header the provider expects the credential in.
    api_key_header: str = "x-api-key"

    @property
    def is_loopback(self) -> bool:
        """Whether this URL points somewhere a credential is not needed."""
        from urllib.parse import urlparse

        host = (urlparse(self.url).hostname or "").lower()
        return host in {"127.0.0.1", "localhost", "::1"} or host.endswith(".test")

    def auth_headers(self) -> dict[str, str]:
        return {self.api_key_header: self.api_key} if self.api_key else {}


@dataclass
class Attempt:
    """What one call to one provider produced."""

    provider: Provider
    outcome: Outcome
    status_code: int | None = None
    body: Any = None
    elapsed_seconds: float = 0.0
    # Kept for logging only. Never serialised into a client response.
    error: BaseException | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome is Outcome.SUCCESS

    @property
    def should_failover(self) -> bool:
        """Whether this failure justifies trying the next provider.

        A 429 or a timeout is about *this* provider's capacity, so another
        provider may well succeed. A 4xx is about the request itself: the same
        malformed body will fail identically everywhere, so failing over would
        burn the fallback's quota to produce the same error twice.
        """
        return self.outcome in (Outcome.RATE_LIMITED, Outcome.TIMEOUT, Outcome.UNAVAILABLE)


def classify(status_code: int) -> Outcome:
    if status_code == 429:
        return Outcome.RATE_LIMITED
    if status_code in (401, 403):
        # Deliberately *not* CLIENT_ERROR. The caller's credential never reaches
        # a provider - it is stripped and replaced with the gateway's own - so a
        # 401 or 403 from upstream is never the caller's fault. It means this
        # gateway's key for this provider is missing, wrong or revoked, which is
        # a per-provider condition another provider may not share.
        #
        # Classifying it as CLIENT_ERROR would suppress failover on the single
        # most likely real-world misconfiguration, and return a 400-series
        # `invalid_request_error` blaming the client for it.
        logger.error(
            "provider returned %d: the gateway's upstream credential for this provider is "
            "missing, invalid or revoked. Failing over.",
            status_code,
        )
        return Outcome.UNAVAILABLE
    if status_code >= 500:
        return Outcome.UNAVAILABLE
    if status_code >= 400:
        return Outcome.CLIENT_ERROR
    return Outcome.SUCCESS


async def call_provider(
    client: httpx.AsyncClient,
    provider: Provider,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> Attempt:
    """Make one attempt, converting every failure mode into an `Attempt`.

    The budget is enforced twice, deliberately:

    * `asyncio.timeout` puts a **wall-clock deadline on the whole attempt**.
      "Times out after 3s" should mean 3s end to end, which is what a caller
      budgeting against this gateway will assume.
    * httpx's own `Timeout` still applies per phase, so a dead socket is noticed
      promptly rather than sitting until the outer deadline.

    The outer deadline is not redundant. httpx's timeouts are per-phase: a
    provider that trickles one byte just inside the read timeout, forever, never
    trips it, and the attempt can outlast its budget without either timeout
    firing. Only a total deadline bounds that.

    One consequence worth stating plainly: a timeout aborts *our* wait, not the
    provider's work. It may still be generating, and still billing. The router
    charges the estimate in that case rather than assuming the request was free.
    """
    payload = {**body, "model": provider.model}
    timeout = httpx.Timeout(provider.timeout_seconds)
    # The gateway's credential is applied last so a relayed client header can
    # never shadow it. `app.py` strips `authorization`/`x-api-key` from the
    # incoming request; this supplies the replacement.
    outbound = {**(headers or {}), **provider.auth_headers()}

    start = _now()
    try:
        async with asyncio.timeout(provider.timeout_seconds):
            response = await client.post(provider.url, json=payload, headers=outbound, timeout=timeout)
    except (TimeoutError, httpx.TimeoutException) as exc:
        log_and_sanitise(f"provider {provider.name} timed out after {provider.timeout_seconds}s", exc)
        return Attempt(provider, Outcome.TIMEOUT, elapsed_seconds=_now() - start, error=exc)
    except httpx.HTTPError as exc:
        log_and_sanitise(f"provider {provider.name} unreachable", exc)
        return Attempt(provider, Outcome.UNAVAILABLE, elapsed_seconds=_now() - start, error=exc)

    elapsed = _now() - start
    outcome = classify(response.status_code)

    try:
        parsed = response.json()
    except ValueError:
        parsed = None

    if outcome is not Outcome.SUCCESS:
        logger.warning(
            "provider %s returned %d after %.3fs (%s)", provider.name, response.status_code, elapsed, outcome
        )

    return Attempt(
        provider=provider,
        outcome=outcome,
        status_code=response.status_code,
        body=parsed,
        elapsed_seconds=elapsed,
    )


@dataclass
class StreamAttempt:
    """A streaming call that has produced response headers but not yet a body.

    The response is held open. `close()` must be called if the caller decides
    not to use it, or the connection leaks.
    """

    provider: Provider
    outcome: Outcome
    status_code: int | None = None
    response: httpx.Response | None = None
    elapsed_seconds: float = 0.0
    error: BaseException | None = None

    @property
    def succeeded(self) -> bool:
        return self.outcome is Outcome.SUCCESS

    @property
    def should_failover(self) -> bool:
        return self.outcome in (Outcome.RATE_LIMITED, Outcome.TIMEOUT, Outcome.UNAVAILABLE)

    async def close(self) -> None:
        if self.response is not None:
            await self.response.aclose()


async def open_provider_stream(
    client: httpx.AsyncClient,
    provider: Provider,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> StreamAttempt:
    """Start a streaming attempt and return once the status line has arrived.

    Failover for a streaming request can only happen *before* any bytes reach
    the client - once the status line has been sent downstream there is no way
    to retract it. Splitting the attempt here is what makes that possible: the
    router sees the provider's status code while it can still choose a
    different provider.

    The timeout covers reaching first byte only. A long generation is not a
    failure, so the read timeout is left off deliberately; the primary's 3s
    budget is a budget for *starting* to answer.
    """
    payload = {**body, "model": provider.model}
    outbound = {**(headers or {}), **provider.auth_headers()}
    timeout = httpx.Timeout(provider.timeout_seconds, read=None)

    start = _now()
    request = client.build_request("POST", provider.url, json=payload, headers=outbound, timeout=timeout)
    try:
        async with asyncio.timeout(provider.timeout_seconds):
            response = await client.send(request, stream=True)
    except (TimeoutError, httpx.TimeoutException) as exc:
        log_and_sanitise(f"provider {provider.name} timed out opening a stream", exc)
        return StreamAttempt(provider, Outcome.TIMEOUT, elapsed_seconds=_now() - start, error=exc)
    except httpx.HTTPError as exc:
        log_and_sanitise(f"provider {provider.name} unreachable", exc)
        return StreamAttempt(provider, Outcome.UNAVAILABLE, elapsed_seconds=_now() - start, error=exc)

    outcome = classify(response.status_code)
    if outcome is not Outcome.SUCCESS:
        logger.warning("provider %s returned %d opening a stream (%s)", provider.name, response.status_code, outcome)
        await response.aclose()
        return StreamAttempt(
            provider, outcome, status_code=response.status_code, elapsed_seconds=_now() - start
        )

    return StreamAttempt(
        provider=provider,
        outcome=Outcome.SUCCESS,
        status_code=response.status_code,
        response=response,
        elapsed_seconds=_now() - start,
    )


def _now() -> float:
    import time

    return time.perf_counter()
