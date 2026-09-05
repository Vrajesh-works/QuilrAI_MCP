"""Gateway configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from llm_router.providers import Provider
from llm_router.ratelimit import DEFAULT_TOKEN_LIMIT, DEFAULT_WINDOW_SECONDS


@dataclass(frozen=True)
class Config:
    database_path: str
    token_limit: int
    window_seconds: float
    primary: Provider
    fallbacks: list[Provider] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> Config:
        primary_url = os.environ.get("LLM_PRIMARY_URL", "http://127.0.0.1:8101/v1/messages")
        fallback_url = os.environ.get("LLM_FALLBACK_URL", "http://127.0.0.1:8102/v1/messages")

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
                )
            ],
        )
