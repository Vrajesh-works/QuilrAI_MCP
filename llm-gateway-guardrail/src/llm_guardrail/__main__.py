"""Run the guardrail: `python -m llm_guardrail`."""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

from llm_guardrail.app import create_app


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LLM_GUARDRAIL_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    uvicorn.run(
        create_app(),
        host=os.environ.get("LLM_GUARDRAIL_HOST", "127.0.0.1"),
        port=int(os.environ.get("LLM_GUARDRAIL_PORT", "8090")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
