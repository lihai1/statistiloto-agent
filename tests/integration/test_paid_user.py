"""Integration test: paid user chat flow with HITL on write tools.

Paid tier → analyst worker → docs+history RAG → mock LLM drafts analysis
→ if LLM plans a write tool (save_numbers) → HITL interrupt → /approve resumes.
Read-only tool calls and plain-text responses proceed without HITL.
"""
import json
import pytest
from langgraph.types import Command

pytestmark = pytest.mark.integration


class TestPaidUserChat:
    def test_paid_user_gets_response(self, client, paid_headers):
        """Paid user sends a message and gets a response (no write tool → no HITL)."""
        resp = client.post(
            "/chat",
            json={"session_id": "paid-sess-1", "message": "Analyze my numbers",
                  "intent": "analyst"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data or "paused" in data

    def test_paid_user_hitl_interrupt_on_write_tool(self, client, paid_headers, mock_llm_store):
        """Paid user analyst flow where LLM plans save_numbers → HITL pause."""
        # Set mock LLM to return a write tool call.
        mock_llm_store.get_llm().responses = [
            "TOOL: save_numbers ARGS: {\"category\": \"lucky\", \"numbers\": [1,2,3,4,5,6]}"
        ]
        resp = client.post(
            "/chat",
            json={"session_id": "paid-hitl-1", "message": "Save my lucky numbers 1,2,3,4,5,6",
                  "intent": "analyst"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should be paused because save_numbers is a write tool
        assert data.get("paused") is True

    def test_paid_user_approve_resumes_write_tool(self, client, paid_headers, mock_llm_store):
        """After HITL pause on write tool, /approve with approved=True resumes."""
        mock_llm_store.get_llm().responses = [
            "TOOL: save_numbers ARGS: {\"category\": \"lucky\", \"numbers\": [1,2,3,4,5,6]}"
        ]
        # First, start the chat to trigger the interrupt
        client.post(
            "/chat",
            json={"session_id": "paid-hitl-2", "message": "Save my lucky numbers 1,2,3,4,5,6",
                  "intent": "analyst"},
            headers=paid_headers,
        )
        # Now approve
        resp = client.post(
            "/approve",
            json={"session_id": "paid-hitl-2", "approved": True},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_paid_user_reject_resumes_write_tool(self, client, paid_headers, mock_llm_store):
        """After HITL pause on write tool, /approve with approved=False returns rejection."""
        mock_llm_store.get_llm().responses = [
            "TOOL: save_numbers ARGS: {\"category\": \"lucky\", \"numbers\": [1,2,3,4,5,6]}"
        ]
        client.post(
            "/chat",
            json={"session_id": "paid-hitl-3", "message": "Save my lucky numbers 1,2,3,4,5,6",
                  "intent": "analyst"},
            headers=paid_headers,
        )
        resp = client.post(
            "/approve",
            json={"session_id": "paid-hitl-3", "approved": False},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "rejected" in (data.get("response") or "").lower()

    def test_paid_user_no_hitl_on_read_tool(self, client, paid_headers, mock_llm_store):
        """Paid user analyst flow where LLM plans a read tool → no HITL, direct response."""
        mock_llm_store.get_llm().responses = [
            "TOOL: get_statistics ARGS: {\"how_many\": 10, \"form_type\": 0}"
        ]
        resp = client.post(
            "/chat",
            json={"session_id": "paid-read-1", "message": "Get statistics",
                  "intent": "analyst"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Read tool → no HITL → should get a response, not paused
        assert "response" in data
        assert data.get("paused") is not True

    def test_paid_user_token_usage_logged(self, client, paid_headers, db_pool, mock_llm_store):
        """Paid user chat logs token usage when the LLM is invoked.

        With the new deterministic architecture, clear requests are resolved
        directly. To verify token logging, we send a save_numbers request
        which routes to the analyst worker (LLM planning + HITL).
        """
        mock_llm_store.get_llm().responses = [
            'TOOL: save_numbers ARGS: {"category": "lucky", "numbers": [1,2,3,4,5,6]}',
            "Numbers saved successfully.",
        ]
        client.post(
            "/chat",
            json={"session_id": "paid-sess-2", "message": "Save my lucky numbers 1,2,3,4,5,6",
                  "intent": "analyst"},
            headers=paid_headers,
        )
        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT user_sub, tier FROM agent.token_usage "
                "WHERE user_sub = 'paid-user-001' LIMIT 1"
            ).fetchone()
        assert row is not None
        assert row[0] == "paid-user-001"
        assert row[1] == "paid"
