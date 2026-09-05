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

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            upstream_url=os.environ.get("LLM_UPSTREAM_URL", "http://127.0.0.1:8091/v1/messages"),
            request_timeout_seconds=float(os.environ.get("LLM_GUARDRAIL_CONNECT_TIMEOUT", "10")),
            read_timeout_seconds=float(os.environ.get("LLM_GUARDRAIL_READ_TIMEOUT", "600")),
        )
