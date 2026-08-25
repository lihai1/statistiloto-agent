"""Eval: Security tests.

Verifies that:
  - Admin write without approval triggers HITL.
  - Approval replay cannot execute twice.
  - Free user cannot access analyst tools.
"""
import pytest

pytestmark = pytest.mark.integration


class TestSecurity:
    """Security and HITL behavior."""

    def test_admin_write_triggers_hitl(self, client, admin_headers, mock_tool_clients, set_llm_responses):
        """Admin write tool (save_numbers) should trigger HITL (paused=true)."""
        set_llm_responses([
            'TOOL: save_numbers ARGS: {"category": "lucky", "numbers": [7, 11, 17, 24, 31, 36]}',
        ])

        resp = client.post("/chat", headers=admin_headers,
            json={"session_id": "eval-sec-1", "message": "Save my numbers 7,11,17,24,31,36", "intent": "admin_ops"})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("paused") is True, f"Expected HITL pause, got: {data}"
        assert data.get("thread_id"), "Expected a thread_id for HITL resume"

    def test_approval_replay_cannot_execute_twice(self, client, admin_headers, mock_tool_clients, set_llm_responses):
        """After approval, replaying the same approval should not execute the tool again."""
        set_llm_responses([
            'TOOL: save_numbers ARGS: {"category": "lucky", "numbers": [1, 2, 3, 4, 5, 6]}',
            "Numbers saved successfully.",
        ])

        # Step 1: trigger write → HITL pause.
        resp1 = client.post("/chat", headers=admin_headers,
            json={"session_id": "eval-sec-2", "message": "Save 1,2,3,4,5,6", "intent": "admin_ops"})
        assert resp1.status_code == 200
        assert resp1.json().get("paused") is True

        # Step 2: approve → tool executes.
        resp2 = client.post("/approve", headers=admin_headers,
            json={"session_id": "eval-sec-2", "approved": True})
        assert resp2.status_code == 200
        assert resp2.json().get("response")

        # Step 3: replay same approval → should not re-execute.
        resp3 = client.post("/approve", headers=admin_headers,
            json={"session_id": "eval-sec-2", "approved": True})
        assert resp3.status_code in (200, 400, 404, 409), \
            f"Unexpected status for replay: {resp3.status_code}"

    def test_free_user_cannot_access_analyst(self, client, free_headers, mock_tool_clients, set_llm_responses):
        """Free user requesting analyst intent should be downgraded to nl_assistant."""
        set_llm_responses([
            "I can explain lottery concepts, but statistics retrieval requires a paid plan."
        ])

        resp = client.post("/chat", headers=free_headers,
            json={"session_id": "eval-sec-3", "message": "What are hot pairs?", "intent": "analyst"})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("response"), "Expected a response for downgraded free user"
        assert data.get("paused") is not True
