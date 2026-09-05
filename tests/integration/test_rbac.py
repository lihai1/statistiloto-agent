"""Integration tests: RBAC matrix — tier-based access control across all endpoints.

Verifies that:
- Free and paid users are rejected (403) from all admin-only endpoints.
- Admin users are accepted (not 401/403) on all admin endpoints.
- All three tiers are accepted on user-level endpoints.
- Missing tokens yield 422 (required Header validation).
- Malformed JWTs yield 401 on both user and admin endpoints.
- Cross-user session isolation: user B cannot see/list/delete user A's sessions.
"""
import pytest

pytestmark = pytest.mark.integration


# ── Endpoint catalogs ────────────────────────────────────────

# All 13 admin-only endpoints (require_admin_user).
# Each entry: (method, path, json_body, query_params)
ADMIN_ENDPOINTS = [
    ("put", "/llm-config", {"provider": "ollama", "model": "llama3.1:8b"}, None),
    ("get", "/llm-configs", None, None),
    ("post", "/llm-configs", {"provider": "ollama", "model": "llama3.1:8b"}, None),
    ("put", "/llm-configs/1", {"provider": "ollama", "model": "llama3.1:8b"}, None),
    ("put", "/llm-configs/1/activate", None, None),
    ("post", "/llm-configs/1/test", None, None),
    ("delete", "/llm-configs/1", None, None),
    ("get", "/llm-models", None, {"provider": "ollama"}),
    ("get", "/free-llm", None, None),
    ("put", "/free-llm", {"enabled": True}, None),
    ("get", "/token-usage", None, None),
    ("get", "/audit-log", None, None),
    ("post", "/reindex", None, None),
]

# User-level endpoints (get_current_user) — any authenticated tier.
# Each entry: (method, path, json_body, query_params)
USER_ENDPOINTS = [
    ("get", "/llm-config", None, None),
    ("get", "/sessions", None, None),
    ("get", "/sessions/rbac-test-sess", None, None),
    ("delete", "/sessions/rbac-test-sess", None, None),
    ("delete", "/sessions", None, None),
]


def _request(client, method, path, headers, json_body=None, params=None):
    """Dispatch a request by HTTP method string."""
    return client.request(
        method,
        path,
        headers=headers,
        json=json_body,
        params=params,
    )


# ── Admin endpoint rejection: free & paid ────────────────────

@pytest.mark.parametrize(
    "method, path, json_body, params",
    ADMIN_ENDPOINTS,
    ids=[f"{m.upper()} {p}" for m, p, _, _ in ADMIN_ENDPOINTS],
)
class TestNonAdminRejectedFromAdminEndpoints:
    def test_free_user_rejected_from_all_admin_endpoints(
        self, client, free_headers, method, path, json_body, params,
    ):
        resp = _request(client, method, path, free_headers, json_body, params)
        assert resp.status_code == 403, (
            f"Free user should be 403 on {method.upper()} {path}, "
            f"got {resp.status_code}: {resp.text}"
        )

    def test_paid_user_rejected_from_all_admin_endpoints(
        self, client, paid_headers, method, path, json_body, params,
    ):
        resp = _request(client, method, path, paid_headers, json_body, params)
        assert resp.status_code == 403, (
            f"Paid user should be 403 on {method.upper()} {path}, "
            f"got {resp.status_code}: {resp.text}"
        )


# ── Admin endpoint acceptance: admin ─────────────────────────

@pytest.mark.parametrize(
    "method, path, json_body, params",
    ADMIN_ENDPOINTS,
    ids=[f"{m.upper()} {p}" for m, p, _, _ in ADMIN_ENDPOINTS],
)
def test_admin_user_accepted_on_all_admin_endpoints(
    client, admin_headers, method, path, json_body, params,
):
    """Admin token must NOT be auth-rejected (401/403) on any admin endpoint.

    Other errors (404 for missing config ids, 400 for active-config delete,
    500 for reindex failures) are acceptable — we only assert the request
    passes the auth gate.
    """
    resp = _request(client, method, path, admin_headers, json_body, params)
    assert resp.status_code not in (401, 403), (
        f"Admin should not be auth-rejected on {method.upper()} {path}, "
        f"got {resp.status_code}: {resp.text}"
    )


# ── User endpoint acceptance: all tiers ──────────────────────

@pytest.mark.parametrize(
    "method, path, json_body, params",
    USER_ENDPOINTS,
    ids=[f"{m.upper()} {p}" for m, p, _, _ in USER_ENDPOINTS],
)
@pytest.mark.parametrize(
    "headers_fixture",
    ["free_headers", "paid_headers", "admin_headers"],
    ids=["free", "paid", "admin"],
)
def test_all_tiers_accepted_on_user_endpoints(
    client, request, headers_fixture, method, path, json_body, params,
):
    """All three tier tokens must NOT be auth-rejected (401/403) on user endpoints."""
    headers = request.getfixturevalue(headers_fixture)
    resp = _request(client, method, path, headers, json_body, params)
    assert resp.status_code not in (401, 403), (
        f"{headers_fixture} should not be auth-rejected on {method.upper()} {path}, "
        f"got {resp.status_code}: {resp.text}"
    )


# ── Missing / invalid token ──────────────────────────────────

def test_missing_token_returns_401_or_422(client):
    """No Authorization header on a user endpoint → 422 (required Header validation)."""
    resp = client.get("/llm-config")
    assert resp.status_code in (401, 422), (
        f"Missing token should be 401 or 422, got {resp.status_code}: {resp.text}"
    )


def test_invalid_token_returns_401(client):
    """Malformed JWT string as Bearer token on a user endpoint → 401."""
    resp = client.get(
        "/llm-config",
        headers={"Authorization": "Bearer not.a.valid.jwt"},
    )
    assert resp.status_code == 401, (
        f"Invalid token should be 401, got {resp.status_code}: {resp.text}"
    )


def test_invalid_token_on_admin_endpoint_returns_401(client):
    """Malformed JWT on an admin endpoint → 401 (NOT 403).

    After the refactor, require_admin_user wraps get_current_user, so an
    invalid token fails at the JWT-validation step (401) before the
    tier check (403) is reached.
    """
    resp = client.get(
        "/llm-configs",
        headers={"Authorization": "Bearer not.a.valid.jwt"},
    )
    assert resp.status_code == 401, (
        f"Invalid token on admin endpoint should be 401, got {resp.status_code}: {resp.text}"
    )


# ── Cross-user session isolation ─────────────────────────────

class TestCrossUserSessionIsolation:
    def test_cross_user_session_isolation(self, client, free_headers, paid_headers):
        """User A creates a chat session; user B cannot list/get/delete it.

        Steps:
          1. User A (free) sends a chat message with session_id="iso-a-1".
          2. User B (paid) lists sessions — user A's session must NOT appear.
          3. User B tries to GET user A's session — must not return A's messages.
          4. User B tries to DELETE user A's session — must get 404.
        """
        # 1. User A creates a session.
        resp_a = client.post(
            "/chat",
            json={"session_id": "iso-a-1", "message": "Generate 5 lottery forms"},
            headers=free_headers,
        )
        assert resp_a.status_code == 200

        # 2. User B lists sessions — user A's session must not be present.
        resp_b_list = client.get("/sessions", headers=paid_headers)
        assert resp_b_list.status_code == 200
        b_sessions = resp_b_list.json().get("sessions", [])
        b_session_ids = {s.get("session_id") for s in b_sessions}
        assert "iso-a-1" not in b_session_ids, (
            f"User B should not see user A's session 'iso-a-1', "
            f"but found it in {b_session_ids}"
        )

        # 3. User B tries to GET user A's session — should not return A's messages.
        resp_b_get = client.get("/sessions/iso-a-1", headers=paid_headers)
        # The session is scoped to user A's sub, so user B gets an empty
        # message list (not a 403, because the endpoint is user-level).
        assert resp_b_get.status_code == 200
        messages = resp_b_get.json().get("messages", [])
        assert messages == [], (
            f"User B should not retrieve user A's session messages, got {messages}"
        )

        # 4. User B tries to DELETE user A's session — must get 404.
        resp_b_del = client.delete("/sessions/iso-a-1", headers=paid_headers)
        assert resp_b_del.status_code == 404, (
            f"User B should get 404 deleting user A's session, "
            f"got {resp_b_del.status_code}: {resp_b_del.text}"
        )
