"""Integration test: SSE streaming chat endpoint.

Verifies that POST /chat/stream returns text/event-stream content type,
emits progress events as graph nodes complete, emits a done event with the
final response, handles HITL pauses, and records the session.
"""
import json

import pytest

pytestmark = pytest.mark.integration


class TestChatStream:
    def test_stream_returns_sse_content_type(self, client, free_headers):
        """POST /chat/stream returns text/event-stream content type."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-1", "message": "Generate 5 lottery forms"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_emits_done_event(self, client, free_headers):
        """SSE output contains event: done with the final response."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-2", "message": "Generate 5 lottery forms"},
            headers=free_headers,
        )
        body = resp.text
        assert "event: done" in body
        # The done event should contain a response field.
        done_lines = [l for l in body.split("\n") if l.startswith("data: ") and "response" in l]
        assert len(done_lines) > 0

    def test_stream_emits_progress_events(self, client, paid_headers):
        """SSE output contains event: progress lines with node names.

        Paid user with an ambiguous message routes to the analyst worker,
        which has multiple graph nodes (retrieve, draft, etc.).
        """
        # Use a message that requires LLM processing (ambiguous analyst path).
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-3", "message": "Analyze my numbers 1 2 3 4 5 6"},
            headers=paid_headers,
        )
        body = resp.text
        # Should have at least one progress event.
        assert "event: progress" in body

    def test_stream_emits_paused_for_hitl(self, client, paid_headers, mock_llm_store):
        """Write tool requests emit event: paused for HITL."""
        # Configure mock LLM to return a write tool call.
        mock_llm_store.get_llm().responses = [
            'TOOL: save_numbers ARGS: {"category": "lucky", "numbers": [1,2,3,4,5,6]}'
        ]
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-4", "message": "Save my lucky numbers 1,2,3,4,5,6",
                  "intent": "analyst"},
            headers=paid_headers,
        )
        body = resp.text
        assert "event: paused" in body
        assert "thread_id" in body

    def test_stream_preserves_existing_chat_behavior(self, client, free_headers):
        """POST /chat (non-streaming) still works unchanged."""
        resp = client.post(
            "/chat",
            json={"session_id": "stream-5", "message": "Generate 5 lottery forms"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data or "paused" in data

    def test_stream_records_session(self, client, free_headers, db_pool):
        """After streaming completes, the session is recorded in chat_sessions."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-6", "message": "Generate 5 lottery forms"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT session_id FROM agent.chat_sessions WHERE session_id = 'stream-6'"
            ).fetchone()
        assert row is not None

    def test_stream_hebrew_response(self, client, free_headers):
        """Streaming with Hebrew input produces a non-empty SSE response.

        This verifies the dicta-instruct-1.7b model works through the
        streaming endpoint with Hebrew input.
        """
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-hebrew", "message": "מה זה מספרים חמים?"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: done" in body


class TestChatStreamRedis:
    """Tests for the Redis pub/sub streaming path."""

    def test_redis_path_returns_json_with_channel(self, client, free_headers, monkeypatch):
        """When Redis is available, /chat/stream returns JSON with thread_id and channel."""
        import asyncio
        from app import redis_client

        # Mock Redis client and availability check.
        class MockRedis:
            def __init__(self):
                self.published = []
                self._set_keys = {}

            async def ping(self):
                return True

            async def publish(self, channel, message):
                self.published.append((channel, message))

            async def setex(self, key, ttl, value):
                self._set_keys[key] = value

            async def delete(self, key):
                self._set_keys.pop(key, None)

        mock_redis = MockRedis()
        monkeypatch.setattr(redis_client, "_redis_client", mock_redis)
        monkeypatch.setattr(redis_client, "_redis_checked", True)

        async def _fake_available():
            return True
        monkeypatch.setattr(redis_client, "is_redis_available", _fake_available)

        resp = client.post(
            "/chat/stream",
            json={"session_id": "redis-1", "message": "Generate 5 lottery forms"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert "channel" in data
        assert data["channel"].startswith("agent:stream:")

        # Reset redis_client state for other tests.
        redis_client.reset_redis_client()

    def test_redis_unavailable_falls_back_to_sse(self, client, free_headers, monkeypatch):
        """When Redis is NOT available, /chat/stream falls back to inline SSE."""
        from app import redis_client

        async def _fake_available():
            return False
        monkeypatch.setattr(redis_client, "is_redis_available", _fake_available)

        resp = client.post(
            "/chat/stream",
            json={"session_id": "redis-fallback-1", "message": "Generate 5 lottery forms"},
            headers=free_headers,
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert "event: done" in body
