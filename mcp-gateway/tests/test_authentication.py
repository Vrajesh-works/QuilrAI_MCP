"""Bearer token handling."""

from __future__ import annotations

import pytest

from conftest import ADMIN_TOKEN, VIEWER_TOKEN, auth
from mcp_gateway.auth import AuthError, extract_bearer_token, resolve_principal

pytestmark = pytest.mark.asyncio

LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


@pytest.mark.parametrize(
    ("header", "why"),
    [
        (None, "no header at all"),
        ("", "empty header"),
        ("admin-token-abc123", "raw token with no scheme"),
        ("Basic YWRtaW46cGFzcw==", "wrong scheme"),
        ("Bearer", "scheme with no token"),
        ("Bearer   ", "scheme with whitespace only"),
        ("Bearer not-a-real-token", "unknown token"),
        ("Bearer admin-token-abc12", "token one character short"),
        ("Bearer ADMIN-TOKEN-ABC123", "token in the wrong case"),
    ],
)
async def test_bad_credentials_are_rejected_with_401(gateway, received, header, why):
    headers = {"Authorization": header} if header is not None else {}
    response = await gateway.post("/mcp", json=LIST, headers=headers)

    assert response.status_code == 401, why
    assert response.json()["error"]["code"] == -32600
    assert received == [], f"unauthenticated request reached downstream ({why})"


async def test_401_carries_www_authenticate(gateway):
    """RFC 9110 §11.6.1 requires the challenge header on a 401."""
    response = await gateway.post("/mcp", json=LIST)
    assert "bearer" in response.headers["www-authenticate"].lower()


async def test_bearer_scheme_is_case_insensitive(gateway):
    """RFC 7235 §2.1: the scheme token is case-insensitive."""
    for scheme in ("Bearer", "bearer", "BEARER", "BeArEr"):
        response = await gateway.post("/mcp", json=LIST, headers={"Authorization": f"{scheme} {VIEWER_TOKEN}"})
        assert response.status_code == 200, scheme


async def test_error_does_not_reveal_whether_the_token_exists(gateway):
    """Distinguishing "unknown token" from "wrong role" turns the endpoint into
    an oracle for enumerating valid tokens."""
    unknown = await gateway.post("/mcp", json=LIST, headers=auth("totally-made-up"))
    near_miss = await gateway.post("/mcp", json=LIST, headers=auth(ADMIN_TOKEN[:-1]))

    assert unknown.json()["error"]["message"] == near_miss.json()["error"]["message"]


async def test_roles_resolve_from_the_token():
    assert resolve_principal(f"Bearer {ADMIN_TOKEN}").role == "admin"
    assert resolve_principal(f"Bearer {ADMIN_TOKEN}").is_admin is True
    assert resolve_principal(f"Bearer {VIEWER_TOKEN}").role == "viewer"
    assert resolve_principal(f"Bearer {VIEWER_TOKEN}").is_admin is False


async def test_extract_bearer_token_strips_the_scheme():
    assert extract_bearer_token("Bearer abc123") == "abc123"
    assert extract_bearer_token("bearer abc123") == "abc123"


async def test_extract_bearer_token_rejects_malformed_headers():
    for header in (None, "", "Bearer", "Token abc", "abc"):
        with pytest.raises(AuthError):
            extract_bearer_token(header)


async def test_health_endpoint_needs_no_auth(gateway):
    response = await gateway.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
