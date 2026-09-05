"""Bearer token -> principal.

A static token table stands in for whatever the real deployment uses (a JWT
verifier, an introspection endpoint, a secrets store). The gateway only needs
one thing from it - the caller's role - so the seam is deliberately narrow and
`resolve_principal` is the single place to swap.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    subject: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


class AuthError(Exception):
    """Authentication failed: no token, malformed header, or unknown token."""


# Demo credentials. Real deployments verify a signed token instead; the shape of
# the result is what the rest of the gateway depends on, not this table.
_DEFAULT_TOKENS: dict[str, Principal] = {
    "admin-token-abc123": Principal(subject="ops@example.com", role="admin"),
    "viewer-token-xyz789": Principal(subject="analyst@example.com", role="viewer"),
    "viewer-token-second": Principal(subject="intern@example.com", role="viewer"),
}


def _token_table() -> dict[str, Principal]:
    """Allow tokens to be supplied out of band, e.g. for tests or a demo.

    Format: `MCP_GATEWAY_TOKENS="token:subject:role,token2:subject2:role2"`.
    """
    configured = os.environ.get("MCP_GATEWAY_TOKENS")
    if not configured:
        return dict(_DEFAULT_TOKENS)

    table: dict[str, Principal] = {}
    for entry in configured.split(","):
        entry = entry.strip()
        if not entry:
            continue
        token, _, rest = entry.partition(":")
        subject, _, role = rest.partition(":")
        if token and subject and role:
            table[token] = Principal(subject=subject, role=role)
    return table


def extract_bearer_token(header_value: str | None) -> str:
    """Pull the token out of an `Authorization` header.

    Raises:
        AuthError: the header is absent or not a well-formed Bearer credential.
    """
    if not header_value:
        raise AuthError("Missing Authorization header.")

    scheme, _, token = header_value.partition(" ")
    # RFC 7235 §2.1: the auth scheme is case-insensitive, so "bearer" is valid.
    if scheme.lower() != "bearer":
        raise AuthError("Authorization header must use the Bearer scheme.")

    token = token.strip()
    if not token:
        raise AuthError("Bearer token is empty.")
    return token


def resolve_principal(header_value: str | None) -> Principal:
    """Authenticate a request from its Authorization header.

    Raises:
        AuthError: authentication failed. The message is safe to return to the
            client; it never distinguishes "unknown token" from "wrong role",
            which would turn the endpoint into a token oracle.
    """
    token = extract_bearer_token(header_value)

    # Compare against every known token rather than a dict lookup, so the work
    # done does not depend on how much of the token matched. A dict hit/miss is
    # a weak timing signal, but constant-time comparison is cheap here.
    matched: Principal | None = None
    for known_token, principal in _token_table().items():
        if hmac.compare_digest(token, known_token):
            matched = principal

    if matched is None:
        raise AuthError("Invalid or unknown bearer token.")
    return matched
