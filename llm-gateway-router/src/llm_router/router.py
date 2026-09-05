"""Routing: admission control, failover, and settling the bill.

The lifecycle of one request:

    reserve quota  ->  try primary  ->  (429 / timeout / 5xx) ->  try fallback
          |                 |                                          |
          |                 +---------------- success ----------------+
          |                                     |
          +------ release on total failure      +--> settle with real usage

Getting the settle step right is what separates this from a toy: quota is held
before the call, corrected afterwards, and given back if nothing was consumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from llm_router import tokens
from llm_router.errors import (
    RATE_LIMITED,
    TIMEOUT,
    UPSTREAM_UNAVAILABLE,
    gateway_error,
)
from llm_router.providers import Attempt, Outcome, Provider, call_provider
from llm_router.ratelimit import LimitDecision, RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """What the caller should send back."""

    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]
    attempts: list[Attempt]

    @property
    def provider_used(self) -> str | None:
        for attempt in self.attempts:
            if attempt.succeeded:
                return attempt.provider.name
        return None


def _rate_limit_headers(decision: LimitDecision) -> dict[str, str]:
    """Standard quota headers, so a well-behaved client can self-throttle."""
    headers = {
        "x-ratelimit-limit-tokens": str(decision.limit),
        "x-ratelimit-remaining-tokens": str(decision.remaining),
    }
    if decision.retry_after_seconds is not None:
        # Retry-After must be an integer number of seconds (RFC 9110 §10.2.3),
        # and rounding up avoids handing back a value that is still too early.
        headers["retry-after"] = str(max(1, int(decision.retry_after_seconds + 0.999)))
    return headers


class Router:
    """Admission control and provider failover for one gateway."""

    def __init__(
        self,
        limiter: RateLimiter,
        primary: Provider,
        fallbacks: list[Provider],
        client: httpx.AsyncClient,
    ):
        self._limiter = limiter
        self._primary = primary
        self._fallbacks = fallbacks
        self._client = client

    @property
    def providers(self) -> list[Provider]:
        return [self._primary, *self._fallbacks]

    async def route(
        self, tenant: str, body: dict[str, Any], headers: dict[str, str] | None = None
    ) -> RouteResult:
        """Admit, dispatch, fail over if needed, and settle the reservation."""
        reservation_size = tokens.reservation_size(body)
        decision = await self._limiter.check_and_reserve(tenant, reservation_size)

        if not decision.allowed:
            logger.info(
                "Rate limited tenant=%s used=%d/%d requested=%d",
                tenant, decision.used_tokens, decision.limit, reservation_size,
            )
            return RouteResult(
                status_code=429,
                body=gateway_error(
                    RATE_LIMITED,
                    detail={
                        "limit_tokens": decision.limit,
                        "used_tokens": decision.used_tokens,
                        "requested_tokens": reservation_size,
                        "window_seconds": self._limiter._window,
                        "retry_after_seconds": round(decision.retry_after_seconds or 0.0, 3),
                    },
                ),
                headers=_rate_limit_headers(decision),
                attempts=[],
            )

        assert decision.reservation is not None
        attempts: list[Attempt] = []

        try:
            for provider in self.providers:
                attempt = await call_provider(self._client, provider, body, headers)
                attempts.append(attempt)

                if attempt.succeeded:
                    await self._settle(decision, attempt, reservation_size)
                    return RouteResult(
                        status_code=200,
                        body=attempt.body if isinstance(attempt.body, dict) else {},
                        headers={
                            **_rate_limit_headers(decision),
                            "x-gateway-provider": provider.name,
                            "x-gateway-attempts": str(len(attempts)),
                        },
                        attempts=attempts,
                    )

                if not attempt.should_failover:
                    # The request itself is at fault; every provider will say the
                    # same thing. Return it rather than burning the fallback.
                    await self._settle(decision, attempt, reservation_size)
                    return self._client_error_result(decision, attempts)

            return await self._exhausted(decision, attempts, reservation_size)

        except BaseException:
            # Nothing was billed for a request that blew up in the gateway.
            await self._limiter.release(decision.reservation)
            raise

    async def _settle(self, decision: LimitDecision, attempt: Attempt, reserved: int) -> None:
        """Correct the reservation to what the request really cost."""
        assert decision.reservation is not None
        actual = tokens.actual_tokens(attempt.body)
        if actual is None:
            # The provider did not report usage. Charging the estimate is the
            # conservative choice: charging zero would make an unreported
            # response a way to bypass the limit entirely.
            actual = reserved
            logger.debug("Provider %s reported no usage; charging the estimate", attempt.provider.name)
        await self._limiter.settle(decision.reservation, actual)

    def _client_error_result(self, decision: LimitDecision, attempts: list[Attempt]) -> RouteResult:
        """Relay a 4xx without echoing the provider's error body.

        The body can name internal deployments, account ids or quota details;
        the status code is the part the caller legitimately needs.
        """
        failed = attempts[-1]
        return RouteResult(
            status_code=failed.status_code or 400,
            body=gateway_error(
                "invalid_request_error",
                detail={"provider_status": failed.status_code, "attempts": len(attempts)},
            ),
            headers=_rate_limit_headers(decision),
            attempts=attempts,
        )

    async def _exhausted(
        self, decision: LimitDecision, attempts: list[Attempt], reserved: int
    ) -> RouteResult:
        """Every provider failed. Release the quota and report it safely."""
        assert decision.reservation is not None

        timed_out = [attempt for attempt in attempts if attempt.outcome is Outcome.TIMEOUT]
        if timed_out:
            # A read timeout does not mean the provider did no work - it may
            # still be generating, and billing for it. Charging the estimate is
            # the safe assumption; releasing it would let a tenant drive
            # unlimited load by timing out every request.
            await self._limiter.settle(decision.reservation, reserved)
        else:
            await self._limiter.release(decision.reservation)

        every_timeout = len(timed_out) == len(attempts)
        error_type = TIMEOUT if every_timeout else UPSTREAM_UNAVAILABLE

        logger.error(
            "All %d provider(s) failed: %s",
            len(attempts),
            ", ".join(f"{attempt.provider.name}={attempt.outcome}" for attempt in attempts),
        )

        return RouteResult(
            status_code=504 if every_timeout else 502,
            body=gateway_error(
                error_type,
                detail={
                    # Provider *names* and outcome categories are gateway-owned
                    # vocabulary, not upstream text, so they are safe to return
                    # and make the failure debuggable from the client side.
                    "attempts": [
                        {
                            "provider": attempt.provider.name,
                            "outcome": str(attempt.outcome),
                            "elapsed_ms": round(attempt.elapsed_seconds * 1000),
                        }
                        for attempt in attempts
                    ]
                },
            ),
            headers=_rate_limit_headers(decision),
            attempts=attempts,
        )
