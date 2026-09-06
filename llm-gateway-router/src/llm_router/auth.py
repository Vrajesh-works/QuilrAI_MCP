"""Tenant API key -> tenant identity.

Why this module exists
----------------------
Treating whatever non-empty string the caller puts in `x-api-key` as the tenant
- no table, no signature check, no lookup - breaks the control this service
exists to provide, in two ways:

* **The limit becomes unenforceable.** Exhaust the 50,000-token window, send
  `x-api-key: anything-else`, get a fresh full window. Repeat forever. "50,000
  tokens per minute per tenant API key" presupposes that the key means
  something.
* **It becomes a cross-tenant denial of service.** Sending a victim's key
  exhausts their quota instead.

Using the raw header value as the `tenant` column also turns a rate-limiting
counter into a credential store: live keys sit in plaintext in `token_usage` on
a persistent volume, indexed, and are logged verbatim on every 429 - precisely
the data class the sibling guardrail service exists to redact.

So a key resolves to an **opaque tenant id**, and only that id is ever passed to
the limiter, written to SQLite, or logged. The credential itself never leaves
this module.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Demo credentials, in the same shape and with the same caveat as the MCP
#: gateway's static table: a real deployment verifies a signed token or calls an
#: introspection endpoint, and `resolve_tenant` is the single place to swap.
DEMO_TENANTS: dict[str, str] = {
    "demo-key-alpha": "tenant-alpha",
    "demo-key-beta": "tenant-beta",
}


class AuthError(Exception):
    """The key was absent, malformed, or unknown."""


@dataclass(frozen=True)
class Tenant:
    """An authenticated caller, identified by something that is not a secret."""

    id: str


def key_fingerprint(api_key: str) -> str:
    """A stable, non-reversible identifier for an unrecognised key.

    Used only when `allow_unauthenticated` is on. It keeps the *isolation*
    property - two different keys still get two different windows - without
    putting the key itself on disk or in a log line.
    """
    return "anon-" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:32]


def parse_tenant_table(configured: str | None) -> dict[str, str]:
    """Parse `LLM_ROUTER_TENANTS="key=tenant-id,key2=tenant-id2"`.

    `=` separates the pair rather than `:`, deliberately. The MCP gateway's
    `token:subject:role` format silently truncates any token containing a colon,
    and provider-shaped keys frequently do.
    """
    if not configured:
        return {}
    table: dict[str, str] = {}
    for entry in configured.split(","):
        api_key, separator, tenant_id = entry.strip().partition("=")
        if separator and api_key.strip() and tenant_id.strip():
            table[api_key.strip()] = tenant_id.strip()
    return table


def extract_api_key(x_api_key: str | None, authorization: str | None) -> str:
    """Pull the credential out of either header the Anthropic SDKs send.

    Raises:
        AuthError: neither header carried a usable value.
    """
    candidate = x_api_key or authorization or ""
    if candidate[:7].lower() == "bearer ":
        candidate = candidate[7:]
    candidate = candidate.strip()
    if not candidate:
        raise AuthError("An API key is required, via x-api-key or Authorization: Bearer.")
    return candidate


def resolve_tenant(
    api_key: str, table: dict[str, str], *, allow_unauthenticated: bool = False
) -> Tenant:
    """Authenticate a key and return the tenant it identifies.

    Raises:
        AuthError: the key is not in the table. The message never distinguishes
            "unknown key" from any other failure, so the endpoint cannot be used
            as an oracle to confirm a guessed key.
    """
    # Constant-time comparison against every entry, on UTF-8 bytes so a key with
    # a non-ASCII character raises AuthError rather than TypeError.
    candidate = api_key.encode("utf-8")
    matched: str | None = None
    for known_key, tenant_id in table.items():
        if hmac.compare_digest(candidate, known_key.encode("utf-8")):
            matched = tenant_id

    if matched is not None:
        return Tenant(id=matched)

    if allow_unauthenticated:
        # Explicitly configured escape hatch. Isolation still holds and the key
        # is still never stored, but the limit is only enforceable against a
        # cooperating client - rotating the header still buys a fresh window.
        return Tenant(id=key_fingerprint(api_key))

    raise AuthError("Invalid or unknown API key.")
