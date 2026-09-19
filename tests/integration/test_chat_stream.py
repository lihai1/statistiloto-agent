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

    # ── Deterministic tool execution tests (bug-1 regression) ──────

    def test_stream_generate_form_deterministic(self, client, paid_headers):
        """Generate form via deterministic route → SSE done with form content."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-gen", "message": "Generate 3 forms"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: done" in body
        # Mock returns forms with numbers [1,2,3,4,5,6] strong 7
        assert "1" in body and "2" in body

    def test_stream_get_statistics_deterministic(self, client, paid_headers):
        """Get statistics via deterministic route → SSE done with statistics."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-stats", "message": "Show me hot pairs"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: done" in body
        # Mock returns groups with count 10
        assert "10" in body

    def test_stream_analyze_deterministic(self, client, paid_headers):
        """Analyze via deterministic route → SSE done with frequency data."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-analyze", "message": "Analyze 1,2,3,4,5,6"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: done" in body

    def test_stream_simulate_deterministic(self, client, paid_headers):
        """Simulate via deterministic route → SSE done with backtest summary."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-sim", "message": "Simulate 1,2,3,4,5,6 with strong 7"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: done" in body
        # Mock returns summary with net -200
        assert "200" in body or "draws" in body.lower()

    def test_stream_vague_generate_clarifies(self, client, paid_headers):
        """Bare 'generate' → clarification asking how many forms."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-vague", "message": "generate"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: done" in body
        # Should ask for clarification (how many forms)
        assert "how many" in body.lower() or "כמה" in body

    def test_stream_compound_not_multi_request(self, client, paid_headers):
        """'explain hot and cold' should NOT trigger multi-request pick-one."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "stream-compound",
                  "message": "explain the methodology behind hot and cold numbers"},
            headers=paid_headers,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "event: done" in body
        # Should NOT contain the multi-request pick-one message
        assert "one request at a time" not in body.lower()


class MockStreamRedis:
    """In-memory Redis Streams stand-in (xadd/expire/xrange semantics)."""

    def __init__(self):
        self.streams = {}  # key -> list of {"data": json_str}
        self.expiry = {}
        self._counter = 0

    async def ping(self):
        return True

    async def xadd(self, key, fields, maxlen=None, approximate=True):
        self._counter += 1
        self.streams.setdefault(key, []).append(dict(fields))
        if maxlen:
            self.streams[key] = self.streams[key][-maxlen:]
        return f"{self._counter}-0"

    async def expire(self, key, ttl):
        self.expiry[key] = ttl

    def entries(self, key):
        """Decode all stream entries as parsed event dicts."""
        return [json.loads(e["data"]) for e in self.streams.get(key, [])]


@pytest.fixture
def mock_stream_redis(monkeypatch):
    """Inject a fake Redis Streams client; patch availability to True."""
    from app import redis_client

    mock = MockStreamRedis()
    monkeypatch.setattr(redis_client, "_redis_client", mock)
    monkeypatch.setattr(redis_client, "_redis_checked", True)

    async def _fake_available():
        return True
    monkeypatch.setattr(redis_client, "is_redis_available", _fake_available)
    yield mock
    redis_client.reset_redis_client()


class TestChatStreamRedis:
    """Tests for the Redis Streams streaming path."""

    def test_redis_path_returns_json_with_channel(self, client, free_headers, mock_stream_redis):
        """With Redis, /chat/stream returns JSON {thread_id, channel} (stream key)."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "redis-1", "message": "Generate 5 lottery forms"},
            headers={**free_headers, "Accept": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "thread_id" in data
        assert "channel" in data
        assert data["channel"].startswith("agent:stream:")

    def test_each_run_gets_a_fresh_stream_key(self, client, free_headers, mock_stream_redis):
        """Two runs on the same session must use distinct stream keys.

        Regression: with a shared `agent:stream:{thread_id}` key, the BFF's
        XREAD from 0-0 on run N replayed run N-1's terminal event (done/
        paused) and ended the relay before the new run's events — the UI
        showed a phantom paused card or a stale response.
        """
        channels = []
        for i in range(2):
            resp = client.post(
                "/chat/stream",
                json={"session_id": "redis-same-session", "message": f"Generate {i + 2} forms"},
                headers={**free_headers, "Accept": "application/json"},
            )
            assert resp.status_code == 200
            channels.append(resp.json()["channel"])
        assert channels[0] != channels[1]
        assert all(c.startswith("agent:stream:") for c in channels)

    def test_redis_stream_receives_all_events(self, client, free_headers, mock_stream_redis):
        """The Redis Stream accumulates progress + exactly one terminal event."""
        import time
        resp = client.post(
            "/chat/stream",
            json={"session_id": "redis-2", "message": "Generate 3 forms"},
            headers={**free_headers, "Accept": "application/json"},
        )
        assert resp.status_code == 200
        key = resp.json()["channel"]
        # Background task publishes asynchronously — wait briefly.
        deadline = time.time() + 10
        events = []
        while time.time() < deadline:
            events = mock_stream_redis.entries(key)
            if any(e.get("event") in ("done", "paused", "error") for e in events):
                break
            time.sleep(0.05)
        assert any(e.get("event") == "done" for e in events), f"no done event: {events}"
        terminal = [e for e in events if e.get("event") in ("done", "paused", "error")]
        assert len(terminal) == 1, f"expected exactly one terminal event: {events}"
        assert mock_stream_redis.expiry.get(key) == 3600

    def test_accept_json_without_redis_returns_503(self, client, free_headers, monkeypatch):
        """Accept: application/json + no Redis → 503 before any graph work."""
        from app import redis_client

        async def _fake_available():
            return False
        monkeypatch.setattr(redis_client, "is_redis_available", _fake_available)

        resp = client.post(
            "/chat/stream",
            json={"session_id": "redis-503", "message": "Generate 3 forms"},
            headers={**free_headers, "Accept": "application/json"},
        )
        assert resp.status_code == 503

    def test_accept_sse_forces_inline(self, client, free_headers, mock_stream_redis):
        """Accept: text/event-stream → inline SSE even when Redis is up."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "redis-sse", "message": "Generate 3 forms"},
            headers={**free_headers, "Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "event: done" in resp.text

    def test_redis_unavailable_falls_back_to_sse(self, client, free_headers, monkeypatch):
        """Default Accept with no Redis → inline SSE fallback."""
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


class TestStreamEventContent:
    """Stream event schema tests: progress labels, tokens, paused action."""

    @staticmethod
    def _parse_sse(body: str):
        """Parse SSE body into a list of (event, data_dict)."""
        events = []
        for block in body.split("\n\n"):
            event = None
            data = None
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        data = line[6:]
            if event:
                events.append((event, data))
        return events

    def test_direct_tool_progress_sequence(self, client, paid_headers):
        """Deterministic direct-tool path emits a single step-started event."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "seq-1", "message": "Generate 3 forms"},
            headers={**paid_headers, "Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        progress_nodes = [d["node"] for e, d in events if e == "progress"]
        assert progress_nodes == ["direct_tool"], f"unexpected progress: {events}"
        terminal = [e for e, _ in events if e in ("done", "paused", "error")]
        assert terminal == ["done"]

    def test_progress_has_labels(self, client, paid_headers):
        """Progress events carry a human-readable label."""
        resp = client.post(
            "/chat/stream",
            json={"session_id": "seq-2", "message": "Generate 3 forms"},
            headers={**paid_headers, "Accept": "text/event-stream"},
        )
        events = self._parse_sse(resp.text)
        for e, d in events:
            if e == "progress":
                assert d.get("label"), f"progress event missing label: {d}"
                assert d["node"] != "__interrupt__"

    def test_paused_event_includes_action(self, client, paid_headers, mock_llm_store):
        """HITL paused event carries the proposed tool and args."""
        mock_llm_store.get_llm().responses = [
            'TOOL: save_numbers ARGS: {"category": "lucky", "numbers": [1,2,3,4,5,6]}'
        ]
        resp = client.post(
            "/chat/stream",
            json={"session_id": "seq-3", "message": "Save my lucky numbers 1,2,3,4,5,6",
                  "intent": "analyst"},
            headers={**paid_headers, "Accept": "text/event-stream"},
        )
        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        paused = [d for e, d in events if e == "paused"]
        assert len(paused) == 1, f"expected one paused event: {events}"
        action = paused[0].get("action") or {}
        assert action.get("tool") == "save_numbers", f"paused action: {paused[0]}"
        assert action.get("args", {}).get("numbers") == [1, 2, 3, 4, 5, 6]

    def test_planner_text_never_leaks_as_token(self, client, paid_headers, mock_llm_store):
        """Planner TOOL: lines must not appear in token events."""
        mock_llm_store.get_llm().responses = [
            'TOOL: get_statistics ARGS: {"how_many": 10, "group_size": 2}',
            "Here are the hot pairs.",
        ]
        resp = client.post(
            "/chat/stream",
            json={"session_id": "seq-4", "message": "show me something interesting",
                  "intent": "analyst"},
            headers={**paid_headers, "Accept": "text/event-stream"},
        )
        events = self._parse_sse(resp.text)
        for e, d in events:
            if e == "token":
                assert "TOOL:" not in (d.get("delta") or "")
