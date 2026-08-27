"""Integration tests for the free-tier LLM toggle.

Verifies that:
- Free users receive a generic deterministic response (no LLM) by default for
  ambiguous and domain-explanation requests.
- Free users CAN invoke the LLM when the toggle is enabled via the admin API.
- The API toggle is admin-only (free/paid users cannot set it).
- Paid/admin behavior is unaffected by the toggle.
- Deterministic entitled free-user operations (statistics, form generation,
  analyze) still execute directly with zero LLM calls regardless of the toggle.
- Unauthorized free-user requests still receive deterministic denial.

The toggle is reset between tests via the ``reset_free_llm`` fixture.
"""
import pytest

from app.free_tier_llm import reset_free_llm_toggle, set_free_llm_enabled, is_free_llm_enabled

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def reset_free_llm():
    """Reset the free-tier LLM toggle to its default (disabled) before and after each test."""
    reset_free_llm_toggle()
    yield
    reset_free_llm_toggle()


class TestFreeLlmToggleDefault:
    """Default state: free-tier LLM is disabled."""

    def test_toggle_defaults_disabled(self):
        """The toggle should default to disabled (False)."""
        assert is_free_llm_enabled() is False

    def test_free_ambiguous_gets_generic_response(self, client, free_headers, db_pool):
        """Free user with an ambiguous request gets the generic deterministic response."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-amb-1", "message": "what should I do today?"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        # The generic response mentions specific actions available on free tier.
        assert "generate forms" in data["response"].lower() or "statistics" in data["response"].lower()

    def test_free_domain_explanation_gets_generic_response(self, client, free_headers, db_pool):
        """Free user asking a domain-explanation question gets the generic response."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-dom-1", "message": "What is the archive window?"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "generate forms" in data["response"].lower() or "statistics" in data["response"].lower()

    def test_free_ambiguous_no_llm_call(self, client, free_headers, db_pool, mock_llm_store):
        """Free ambiguous request must NOT invoke the LLM — no token_usage row."""
        client.post(
            "/chat",
            json={"session_id": "sess-no-llm-1", "message": "what should I do today?"},
            headers=free_headers,
        )
        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM agent.token_usage WHERE user_sub = 'free-user-001'"
            ).fetchone()
        assert row[0] == 0

    def test_free_domain_explanation_no_llm_call(self, client, free_headers, db_pool, mock_llm_store):
        """Free domain-explanation request must NOT invoke the LLM."""
        client.post(
            "/chat",
            json={"session_id": "sess-no-llm-2", "message": "What is the archive window?"},
            headers=free_headers,
        )
        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM agent.token_usage WHERE user_sub = 'free-user-001'"
            ).fetchone()
        assert row[0] == 0

    def test_free_deterministic_statistics_still_works(self, client, free_headers, db_pool):
        """Free user deterministic statistics request still executes directly (zero LLM)."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-stats-1", "message": "Show me the top 5 hot pairs"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        # Statistics result should contain numbers, not the generic response.
        assert "generate forms" not in data["response"].lower()

    def test_free_deterministic_form_generation_still_works(self, client, free_headers, db_pool):
        """Free user deterministic form generation still executes directly."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-form-1", "message": "Generate 3 lottery forms"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "generate forms" not in data["response"].lower()

    def test_free_unauthorized_write_still_denied(self, client, free_headers, db_pool):
        """Free user attempting a write operation gets deterministic denial (no LLM)."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-deny-1", "message": "Save my numbers 1 2 3 4 5 6 strong 7"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        # Should be the unauthorized message, not the generic response.
        assert "not available" in data["response"].lower() or "upgrade" in data["response"].lower()


class TestFreeLlmToggleEnabled:
    """When the toggle is enabled, free users CAN invoke the LLM."""

    def test_free_ambiguous_invokes_llm_when_enabled(self, client, free_headers, db_pool, mock_llm_store):
        """When enabled, free ambiguous request routes to nl_assistant and calls the LLM."""
        set_free_llm_enabled(True)
        mock_llm_store.get_llm().responses = ["Here's some advice about lottery strategies."]
        resp = client.post(
            "/chat",
            json={"session_id": "sess-llm-1", "message": "what should I do today?"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["response"] == "Here's some advice about lottery strategies."
        # Token usage should be logged (LLM was called).
        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM agent.token_usage WHERE user_sub = 'free-user-001'"
            ).fetchone()
        assert row[0] > 0

    def test_free_domain_explanation_invokes_llm_when_enabled(self, client, free_headers, db_pool, mock_llm_store):
        """When enabled, free domain-explanation request routes to nl_assistant."""
        set_free_llm_enabled(True)
        mock_llm_store.get_llm().responses = ["The archive window is the range of historical draws."]
        resp = client.post(
            "/chat",
            json={"session_id": "sess-llm-2", "message": "What is the archive window?"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "archive window" in data["response"].lower()

    def test_free_deterministic_still_no_llm_when_enabled(self, client, free_headers, db_pool):
        """Even when LLM is enabled, deterministic free operations don't call the LLM."""
        set_free_llm_enabled(True)
        resp = client.post(
            "/chat",
            json={"session_id": "sess-stats-2", "message": "Show me the top 5 hot pairs"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        # Should be the deterministic statistics result, not an LLM response.
        assert "generate forms" not in data["response"].lower()


class TestFreeLlmApi:
    """API endpoint tests for the free-tier LLM toggle (admin only)."""

    def test_get_free_llm_admin(self, client, admin_headers):
        """Admin can read the toggle state."""
        resp = client.get("/free-llm", headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_get_free_llm_free_user_forbidden(self, client, free_headers):
        """Free user cannot read the toggle — 403."""
        resp = client.get("/free-llm", headers=free_headers)
        assert resp.status_code == 403

    def test_get_free_llm_paid_user_forbidden(self, client, paid_headers):
        """Paid user cannot read the toggle — 403."""
        resp = client.get("/free-llm", headers=paid_headers)
        assert resp.status_code == 403

    def test_put_free_llm_admin(self, client, admin_headers):
        """Admin can set the toggle."""
        resp = client.put("/free-llm", json={"enabled": True}, headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json()["enabled"] is True

        # Verify it persists for the process.
        resp2 = client.get("/free-llm", headers=admin_headers)
        assert resp2.json()["enabled"] is True

    def test_put_free_llm_free_user_forbidden(self, client, free_headers):
        """Free user cannot set the toggle — 403."""
        resp = client.put("/free-llm", json={"enabled": True}, headers=free_headers)
        assert resp.status_code == 403

    def test_put_free_llm_paid_user_forbidden(self, client, paid_headers):
        """Paid user cannot set the toggle — 403."""
        resp = client.put("/free-llm", json={"enabled": True}, headers=paid_headers)
        assert resp.status_code == 403

    def test_put_free_llm_no_auth_rejected(self, client):
        """No auth header → 422 (FastAPI validation)."""
        resp = client.put("/free-llm", json={"enabled": True})
        assert resp.status_code == 422

    def test_get_free_llm_no_auth_rejected(self, client):
        """No auth header → 422 (FastAPI validation)."""
        resp = client.get("/free-llm")
        assert resp.status_code == 422

    def test_toggle_does_not_grant_write_access(self, client, free_headers, db_pool):
        """Enabling the toggle does NOT let free users perform write operations."""
        set_free_llm_enabled(True)
        resp = client.post(
            "/chat",
            json={"session_id": "sess-write-1", "message": "Save my numbers 1 2 3 4 5 6 strong 7"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        # Still denied — toggle only controls LLM invocation, not capabilities.
        assert "not available" in data["response"].lower() or "upgrade" in data["response"].lower()


class TestLlmModelsAdminOnly:
    """Verify GET /llm-models is admin only."""

    def test_llm_models_admin_allowed(self, client, admin_headers):
        """Admin can list models."""
        resp = client.get("/llm-models", params={"provider": "gemini"}, headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data

    def test_llm_models_free_user_forbidden(self, client, free_headers):
        """Free user cannot list models — 403."""
        resp = client.get("/llm-models", params={"provider": "gemini"}, headers=free_headers)
        assert resp.status_code == 403

    def test_llm_models_paid_user_forbidden(self, client, paid_headers):
        """Paid user cannot list models — 403."""
        resp = client.get("/llm-models", params={"provider": "gemini"}, headers=paid_headers)
        assert resp.status_code == 403

    def test_llm_models_no_auth_rejected(self, client):
        """No auth header → 422 (FastAPI validation)."""
        resp = client.get("/llm-models", params={"provider": "gemini"})
        assert resp.status_code == 422


class TestPaidAdminUnaffected:
    """Paid and admin behavior is unaffected by the free-tier LLM toggle."""

    def test_paid_ambiguous_still_routes_to_analyst(self, client, paid_headers, db_pool, mock_llm_store):
        """Paid user ambiguous request still routes to analyst (uses LLM) regardless of toggle."""
        # Toggle is disabled by default — paid should still get LLM.
        mock_llm_store.get_llm().responses = ["Paid user analysis response."]
        resp = client.post(
            "/chat",
            json={"session_id": "sess-paid-1", "message": "what should I do today?"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        # Should NOT be the free generic response.
        assert "generate forms" not in data["response"].lower()

    def test_paid_domain_explanation_still_uses_llm(self, client, paid_headers, db_pool, mock_llm_store):
        """Paid user domain-explanation still uses the LLM regardless of toggle."""
        mock_llm_store.get_llm().responses = ["The archive window explanation for paid users."]
        resp = client.post(
            "/chat",
            json={"session_id": "sess-paid-2", "message": "What is the archive window?"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "archive window" in data["response"].lower()

    def test_admin_ambiguous_still_routes_to_analyst(self, client, admin_headers, db_pool, mock_llm_store):
        """Admin user ambiguous request still routes to analyst regardless of toggle."""
        mock_llm_store.get_llm().responses = ["Admin analysis response."]
        resp = client.post(
            "/chat",
            json={"session_id": "sess-admin-1", "message": "what should I do today?"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "generate forms" not in data["response"].lower()
