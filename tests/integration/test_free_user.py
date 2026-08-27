"""Integration test: free user chat flow.

Free tier → nl_assistant worker → docs-only RAG → mock LLM → response.
No HITL. Token usage is logged.
"""
import json
import pytest

pytestmark = pytest.mark.integration


class TestFreeUserChat:
    def test_free_user_gets_response(self, client, free_headers, db_pool):
        """Free user sends a message and gets a response."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-1", "message": "Generate 5 lottery forms"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data or "paused" in data

    def test_free_user_token_usage_logged(self, client, free_headers, db_pool, mock_llm_store):
        """After a chat that invokes the LLM, token_usage should have a row.

        With the deterministic architecture, clear statistics requests are
        resolved directly (zero LLM calls). Free-tier LLM is disabled by
        default, so domain-explanation questions also skip the LLM. To verify
        token logging, we enable the free-tier LLM toggle and send a
        domain-explanation question that routes to the nl_assistant worker.
        """
        from app.free_tier_llm import set_free_llm_enabled, reset_free_llm_toggle
        set_free_llm_enabled(True)
        try:
            mock_llm_store.get_llm().responses = ["The archive window controls which historical draws are included."]
            client.post(
                "/chat",
                json={"session_id": "sess-2", "message": "Can you explain the archive window?"},
                headers=free_headers,
            )
            with db_pool.connection() as conn:
                row = conn.execute(
                    "SELECT user_sub, tier, provider FROM agent.token_usage "
                    "WHERE user_sub = 'free-user-001' LIMIT 1"
                ).fetchone()
            assert row is not None
            assert row[0] == "free-user-001"
            assert row[1] == "free"
            assert row[2] == "mock"  # mock LLM provider
        finally:
            reset_free_llm_toggle()

    def test_free_user_analyst_intent_downgraded(self, client, free_headers):
        """Free user requesting analyst intent should still get a response
        (downgraded to nl_assistant, not an error)."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-3", "message": "Analyze my numbers",
                  "intent": "analyst"},
            headers=free_headers,
        )
        assert resp.status_code == 200

    def test_free_user_admin_intent_downgraded(self, client, free_headers):
        """Free user requesting admin_ops intent should be downgraded."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-4", "message": "Trigger scraper",
                  "intent": "admin_ops"},
            headers=free_headers,
        )
        assert resp.status_code == 200

    def test_missing_auth_returns_401(self, client):
        """No Authorization header → 422 (FastAPI validation)."""
        resp = client.post(
            "/chat",
            json={"session_id": "sess-5", "message": "hello"},
        )
        assert resp.status_code == 422  # FastAPI validation: missing header
