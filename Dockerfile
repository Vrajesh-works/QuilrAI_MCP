# One image for every service in the stack.
#
# The uv workspace makes this possible: a single `uv sync` installs all seven
# packages (four projects plus their mock upstreams) into one environment, so
# the services differ only by the command compose gives them. Per-project
# images would mean four builds of an almost identical dependency set.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: manifests change far less often than
# source, so editing a .py file does not re-resolve and re-download everything.
# The workspace members' own pyproject files are part of the resolution, so they
# have to be present before `uv sync` runs.
COPY pyproject.toml uv.lock ./
COPY mcp-server/pyproject.toml            mcp-server/pyproject.toml
COPY mcp-gateway/pyproject.toml           mcp-gateway/pyproject.toml
COPY llm-gateway-guardrail/pyproject.toml llm-gateway-guardrail/pyproject.toml
COPY llm-gateway-router/pyproject.toml    llm-gateway-router/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace

# `--no-install-workspace` above skipped the members themselves; this sync
# installs them now that their source is present.
COPY mcp-server/            mcp-server/
COPY mcp-gateway/           mcp-gateway/
COPY llm-gateway-guardrail/ llm-gateway-guardrail/
COPY llm-gateway-router/    llm-gateway-router/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen


FROM python:3.13-slim-bookworm AS runtime

# Non-root: these are network-facing services, and a gateway is precisely the
# kind of process an attacker reaches first.
RUN useradd --create-home --uid 10001 gateway

WORKDIR /app
COPY --from=builder --chown=gateway:gateway /app /app

# Use the venv's interpreter directly; no `uv run`, no activation.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Containers must listen on all interfaces, not the 127.0.0.1 that is the right
# default when running these on a laptop.
ENV MOCK_HOST=0.0.0.0 \
    MCP_GATEWAY_HOST=0.0.0.0 \
    LLM_GUARDRAIL_HOST=0.0.0.0 \
    LLM_ROUTER_HOST=0.0.0.0

# The router keeps its sliding window on disk; compose mounts a volume here.
RUN mkdir -p /data && chown gateway:gateway /data
ENV LLM_ROUTER_DB=/data/router.sqlite

USER gateway

# Overridden per service in compose.yaml.
CMD ["python", "-m", "mcp_gateway"]
