"""Gateway configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    upstream_url: str
    request_timeout_seconds: float
    # Read timeout is separate and generous: a streaming response is idle
    # between tokens, and a single timeout covering the whole body would abort
    # long generations that are behaving perfectly.
    read_timeout_seconds: float
    #: Bearer tokens permitted to use this proxy. See `trust boundary` below.
    tokens: frozenset[str] = frozenset()
    #: When true, the proxy serves unauthenticated callers. Only safe where
    #: something else terminates authentication in front of it.
    trusted_network: bool = True
    #: A prompt is text, not a file upload.
    max_body_bytes: int = 4_194_304

    @property
    def requires_authentication(self) -> bool:
        return bool(self.tokens)

    def describe_trust_boundary(self) -> str:
        """One line for the startup log, so the deployment posture is on record.

        The trust boundary here is a deliberate decision. This proxy does not
        terminate authentication: it relays the *caller's own* provider
        credential to a fixed, environment-configured upstream. There is no
        SSRF - the URL is never caller-controlled - but anyone who can reach the
        port can use it as an open relay to the configured provider, on their
        own credential.

        That is defensible for an internal sidecar behind an authenticating
        gateway. It is not defensible on a reachable port, so
        `LLM_GUARDRAIL_TOKENS` exists to close it, and either way the choice is
        logged at startup rather than left implicit.
        """
        if self.requires_authentication:
            return f"authenticated: {len(self.tokens)} bearer token(s) accepted"
        return (
            "UNAUTHENTICATED. This proxy relays the caller's own provider credential to "
            f"{self.upstream_url} and does not check who is calling. Anyone who can reach "
            "this port can use it as a relay. Deploy it on a private network behind an "
            "authenticating gateway, or set LLM_GUARDRAIL_TOKENS to require a bearer token."
        )

    @classmethod
    def from_env(cls) -> Config:
        raw_tokens = os.environ.get("LLM_GUARDRAIL_TOKENS", "")
        tokens = frozenset(token.strip() for token in raw_tokens.split(",") if token.strip())
        return cls(
            upstream_url=os.environ.get("LLM_UPSTREAM_URL", "http://127.0.0.1:8091/v1/messages"),
            request_timeout_seconds=float(os.environ.get("LLM_GUARDRAIL_CONNECT_TIMEOUT", "10")),
            read_timeout_seconds=float(os.environ.get("LLM_GUARDRAIL_READ_TIMEOUT", "600")),
            tokens=tokens,
            trusted_network=not tokens,
            max_body_bytes=int(os.environ.get("LLM_GUARDRAIL_MAX_BODY_BYTES", "4194304")),
        )
