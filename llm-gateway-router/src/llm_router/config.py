"""Gateway configuration, with validation that happens at boot rather than
per-request.

Checking configuration only when a request needs it means a missing provider
credential surfaces in production, one request at a time, as a 400-series error
blaming the client. `validate()` runs in the app's lifespan and refuses to start
instead.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from llm_router.auth import DEMO_TENANTS, parse_tenant_table
from llm_router.providers import Provider
from llm_router.ratelimit import DEFAULT_TOKEN_LIMIT, DEFAULT_WINDOW_SECONDS

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """The configuration cannot safely be served."""


@dataclass(frozen=True)
class Config:
    database_path: str
    token_limit: int
    window_seconds: float
    primary: Provider
    fallbacks: list[Provider] = field(default_factory=list)
    #: API key -> opaque tenant id. Never used as a rate-limit key itself.
    tenants: dict[str, str] = field(default_factory=lambda: dict(DEMO_TENANTS))
    #: When on, an unrecognised key is admitted under a hash of itself instead
    #: of being rejected. Isolation survives; enforceability does not.
    allow_unauthenticated_tenants: bool = False

    @property
    def providers(self) -> list[Provider]:
        return [self.primary, *self.fallbacks]

    def validate(self) -> None:
        """Refuse to serve a configuration that cannot work or cannot be safe.

        Raises:
            ConfigError: with a message that names the environment variable to
                set. A startup failure that does not say what to do is only
                marginally better than failing per-request.
        """
        for provider in self.providers:
            if not provider.is_loopback and not provider.api_key:
                raise ConfigError(
                    f"Provider {provider.name!r} points at {provider.url!r}, which is not loopback, "
                    f"but no upstream credential is configured. Set "
                    f"LLM_{provider.name.upper()}_API_KEY, or point it at a local mock."
                )

        if not self.tenants and not self.allow_unauthenticated_tenants:
            raise ConfigError(
                "No tenant API keys are configured and unauthenticated tenants are not allowed, "
                "so every request would be rejected. Set LLM_ROUTER_TENANTS='key=tenant-id,...' "
                "or LLM_ROUTER_ALLOW_UNAUTHENTICATED=1."
            )

        if self.allow_unauthenticated_tenants:
            logger.warning(
                "LLM_ROUTER_ALLOW_UNAUTHENTICATED is set. Any API key is accepted, so the "
                "per-tenant token limit is enforceable only against a cooperating client: "
                "rotating the header buys a fresh window."
            )

        if self.tenants == DEMO_TENANTS:
            logger.warning(
                "Using the built-in demo tenant table. Set LLM_ROUTER_TENANTS before serving "
                "anything real."
            )

        if self.token_limit <= 0 or self.window_seconds <= 0:
            raise ConfigError("LLM_ROUTER_TOKEN_LIMIT and LLM_ROUTER_WINDOW_SECONDS must be positive.")

    @classmethod
    def from_env(cls) -> Config:
        primary_url = os.environ.get("LLM_PRIMARY_URL", "http://127.0.0.1:8101/v1/messages")
        fallback_url = os.environ.get("LLM_FALLBACK_URL", "http://127.0.0.1:8102/v1/messages")
        configured_tenants = parse_tenant_table(os.environ.get("LLM_ROUTER_TENANTS"))

        return cls(
            # On disk, not in memory: the window must survive a restart, or
            # bouncing the gateway would reset every tenant's quota.
            database_path=os.environ.get("LLM_ROUTER_DB", "./data/router.sqlite"),
            token_limit=int(os.environ.get("LLM_ROUTER_TOKEN_LIMIT", DEFAULT_TOKEN_LIMIT)),
            window_seconds=float(os.environ.get("LLM_ROUTER_WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS)),
            primary=Provider(
                name="primary",
                url=primary_url,
                model=os.environ.get("LLM_PRIMARY_MODEL", "claude-opus-5"),
                timeout_seconds=float(os.environ.get("LLM_PRIMARY_TIMEOUT", "3.0")),
                # Read from the environment at startup, never from a request
                # header, so a caller cannot choose the upstream credential.
                api_key=os.environ.get("LLM_PRIMARY_API_KEY") or None,
                api_key_header=os.environ.get("LLM_PRIMARY_API_KEY_HEADER", "x-api-key"),
            ),
            fallbacks=[
                Provider(
                    name="fallback",
                    url=fallback_url,
                    model=os.environ.get("LLM_FALLBACK_MODEL", "claude-sonnet-5"),
                    # Longer: having already spent the primary's budget, giving
                    # the backup the same 3s risks failing a request that was
                    # about to succeed.
                    timeout_seconds=float(os.environ.get("LLM_FALLBACK_TIMEOUT", "10.0")),
                    api_key=os.environ.get("LLM_FALLBACK_API_KEY") or None,
                    api_key_header=os.environ.get("LLM_FALLBACK_API_KEY_HEADER", "x-api-key"),
                )
            ],
            tenants=configured_tenants or dict(DEMO_TENANTS),
            allow_unauthenticated_tenants=os.environ.get("LLM_ROUTER_ALLOW_UNAUTHENTICATED", "").lower()
            in {"1", "true", "yes"},
        )
