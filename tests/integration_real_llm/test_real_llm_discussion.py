"""Real LLM discussion integration tests.

Runs the FastAPI app against the real Ollama qwen2.5:0.5b model.
These tests are intentionally observational: they capture the actual
responses and assert that the assistant is helpful, uses tools correctly,
and gates write tools with HITL.
"""

from __future__ import annotations

import json
import time

import pytest

from app.rag.indexers import index_document
from app.security import create_test_jwt

pytestmark = [pytest.mark.integration]


def _real_embedding(text: str):
    """Compute a real nomic-embed-text vector for a short text."""
    from langchain_ollama import OllamaEmbeddings
    from app.config.settings import get_settings

    s = get_settings()
    emb = OllamaEmbeddings(model=s.rag.embedding_model, base_url=s.llm.ollama.base_url)
    return emb.embed_query(text)


@pytest.fixture
def seeded_docs():
    """Seed the docs corpus with a few real chunks so RAG has context."""
    docs = [
        (
            "Statistiloto is a lottery statistics and number-generation assistant. "
            "It helps users generate lottery forms, view historical statistics, "
            "and manage saved numbers.",
            {"source": "product-docs"},
        ),
        (
            "The most frequent pairs are computed by scanning the historical draw "
            "database and counting how many times each pair of numbers has appeared together.",
            {"source": "lottery-guide"},
        ),
        (
            "Paid users can save their favorite numbers and request detailed "
            "multi-step analysis. Admin users can view token usage and trigger scrapers.",
            {"source": "tier-guide"},
        ),
    ]
    for content, metadata in docs:
        index_document("docs", content, metadata, _real_embedding(content))
    yield
    # Keep docs around so multiple tests can see them; real DB is not wiped.


class TestRealLLMDiscussion:
    def test_free_user_question_with_rag(self, client, free_headers, seeded_docs):
        """Free user asks a product question; the assistant should answer from docs."""
        resp = client.post(
            "/chat",
            json={
                "session_id": f"real-free-{int(time.time())}",
                "message": "What is Statistiloto and what can I do with it?",
            },
            headers=free_headers,
        )
        print(f"\n[FREE USER] status={resp.status_code} body={resp.text[:500]}")
        assert resp.status_code == 200
        data = resp.json()
        response = data.get("response") or ""
        assert response, "Free user should receive a non-empty response"
        # The small model may not repeat the exact name, but it should be relevant.
        assert "lottery" in response.lower(), f"Expected lottery context, got: {response[:200]}"

    def test_paid_user_read_tool_for_statistics(self, client, paid_headers, seeded_docs):
        """Paid user asks for frequent pairs; analyst should call get_statistics read tool."""
        session_id = f"real-paid-stats-{int(time.time())}"
        resp = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "Show me the 5 most frequent pairs for the standard lottery.",
                "intent": "analyst",
            },
            headers=paid_headers,
        )
        print(f"\n[PAID STATS] status={resp.status_code} body={resp.text[:1000]}")
        assert resp.status_code == 200
        data = resp.json()

        # If the model produced a write tool or malformed tool it may pause.
        if data.get("paused"):
            pytest.skip(f"Model produced a write/malformed tool, pausing: {data}")

        response = data.get("response") or ""
        assert response, "Paid user should receive a non-empty response"
        # We are lenient because a 0.5b model may not use the exact tool format,
        # but if it does use get_statistics, the mocked tool returns a "pairs" key.
        # The current implementation is too brittle for small models; this test
        # captures the actual output for later analysis.
        if "pairs" not in response.lower() and "count" not in response.lower():
            pytest.fail(
                f"Analyst did not surface statistics result. Actual response:\n{response}"
            )

    def test_paid_user_write_tool_triggers_hitl(self, client, paid_headers, seeded_docs):
        """Paid user asks to save numbers; analyst should pause for HITL approval."""
        session_id = f"real-paid-save-{int(time.time())}"
        resp = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "Save the numbers 1, 2, 3, 4, 5, 6 under category 'lucky'.",
                "intent": "analyst",
            },
            headers=paid_headers,
        )
        print(f"\n[PAID SAVE] status={resp.status_code} body={resp.text[:1000]}")
        assert resp.status_code == 200
        data = resp.json()

        # We expect the graph to pause for HITL if the LLM understood the request.
        if data.get("paused"):
            # Approve and verify the save completes.
            resp2 = client.post(
                "/approve",
                json={"session_id": session_id, "approved": True},
                headers=paid_headers,
            )
            print(f"[PAID SAVE APPROVE] status={resp2.status_code} body={resp2.text[:500]}")
            assert resp2.status_code == 200
            assert "saved" in (resp2.json().get("response") or "").lower()
            return

        # If it did not pause, the model likely returned free text or a malformed tool call.
        # We record the actual response so the developer can improve the prompt/parse logic.
        actual = data.get("response") or ""
        pytest.fail(
            f"Expected HITL pause for save_numbers, got response:\n{actual}"
        )

    def test_admin_user_read_tool(self, client, admin_headers, seeded_docs):
        """Admin asks for token usage; admin_ops should call read_token_usage read tool."""
        session_id = f"real-admin-tokens-{int(time.time())}"
        resp = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "Show me the recent token usage for all users.",
                "intent": "admin_ops",
            },
            headers=admin_headers,
        )
        print(f"\n[ADMIN TOKENS] status={resp.status_code} body={resp.text[:1000]}")
        assert resp.status_code == 200
        data = resp.json()

        if data.get("paused"):
            pytest.skip(f"Admin flow paused: {data}")

        response = data.get("response") or ""
        assert response, "Admin should receive a non-empty response"
        # The tool returns a list of rows; the response should mention at least
        # the structure or some user/provider data.
        if "token" not in response.lower() and "usage" not in response.lower():
            pytest.fail(
                f"Admin did not surface token-usage result. Actual response:\n{response}"
            )

    def test_multi_turn_conversation_remembers_history(self, client, paid_headers, seeded_docs):
        """Two-turn conversation: the assistant should remember the first turn."""
        session_id = f"real-multi-{int(time.time())}"
        first = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "My favorite number is 7.",
                "intent": "nl_assistant",
            },
            headers=paid_headers,
        )
        assert first.status_code == 200
        print(f"\n[MULTI-1] body={first.text[:500]}")

        second = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "What is my favorite number?",
                "intent": "nl_assistant",
            },
            headers=paid_headers,
        )
        print(f"[MULTI-2] body={second.text[:500]}")
        assert second.status_code == 200
        response = second.json().get("response") or ""
        assert "7" in response, f"Assistant should remember favorite number 7. Got: {response[:500]}"
