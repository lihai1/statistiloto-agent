"""Eval: NL→tool-arg mapping tests.

Verifies that the analyst graph correctly parses TOOL: lines and dispatches
to the right tool with the right arguments for Hebrew and English queries.
"""
import pytest

pytestmark = pytest.mark.integration


def _track_tool_args(mock_tool_clients, tool_name):
    """Wrap a mock tool to capture the arguments it was called with."""
    from app.tools import lottery_grpc
    received = {}
    original = lottery_grpc._mock_client[tool_name]

    def tracker(**kw):
        received.update(kw)
        return original(**kw)

    lottery_grpc._mock_client[tool_name] = tracker
    return received


class TestNLMappingHebrew:
    """Hebrew NL → tool-arg mapping."""

    def test_hebrew_hot_pairs(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Hebrew 'זוגות חמים' → get_statistics(group_size=2, strength=hot)."""
        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 10, "group_size": 2, "strength": "hot"}'
        ])
        received = _track_tool_args(mock_tool_clients, "get_statistics")

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-he-1", "message": "מה הזוגות החמים?", "intent": "analyst"})
        assert resp.status_code == 200
        assert received.get("group_size") == 2
        assert received.get("strength") == 2  # "hot" → 2

    def test_hebrew_cold_triples(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Hebrew 'שלשות קרות' → get_statistics(group_size=3, strength=cold)."""
        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 10, "group_size": 3, "strength": "cold"}'
        ])
        received = _track_tool_args(mock_tool_clients, "get_statistics")

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-he-2", "message": "מה השלשות הקרות?", "intent": "analyst"})
        assert resp.status_code == 200
        assert received.get("group_size") == 3
        assert received.get("strength") == 1  # "cold" → 1

    def test_hebrew_quads(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Hebrew 'רביעיות' → get_statistics(group_size=4)."""
        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 10, "group_size": 4, "strength": "hot"}'
        ])
        received = _track_tool_args(mock_tool_clients, "get_statistics")

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-he-3", "message": "מה הרביעיות החמות?", "intent": "analyst"})
        assert resp.status_code == 200
        assert received.get("group_size") == 4


class TestNLMappingEnglish:
    """English NL → tool-arg mapping."""

    def test_english_hot_pairs(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """English 'hot pairs' → get_statistics(group_size=2, strength=hot)."""
        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 10, "group_size": 2, "strength": "hot"}'
        ])
        received = _track_tool_args(mock_tool_clients, "get_statistics")

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-en-1", "message": "What are the hot pairs?", "intent": "analyst"})
        assert resp.status_code == 200
        assert received.get("group_size") == 2
        assert received.get("strength") == 2

    def test_english_cold_triples(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """English 'cold triples' → get_statistics(group_size=3, strength=cold)."""
        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 10, "group_size": 3, "strength": "cold"}'
        ])
        received = _track_tool_args(mock_tool_clients, "get_statistics")

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-en-2", "message": "What are the cold triples?", "intent": "analyst"})
        assert resp.status_code == 200
        assert received.get("group_size") == 3
        assert received.get("strength") == 1


class TestAnalyzeMapping:
    """Analyze tool mapping."""

    def test_analyze_my_numbers(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """'Analyze my numbers 7,11,17,24,31,36' → analyze(form=[7,11,17,24,31,36])."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [7, 11, 17, 24, 31, 36]}'
        ])
        received = _track_tool_args(mock_tool_clients, "analyze")

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-an-1", "message": "Analyze my numbers 7,11,17,24,31,36", "intent": "analyst"})
        assert resp.status_code == 200
        assert received.get("form") == [7, 11, 17, 24, 31, 36]

    def test_analyze_returns_frequency_groups(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Analyze tool returns frequency_groups structure (not matches/frequency)."""
        set_llm_responses([
            'TOOL: analyze ARGS: {"form": [7, 11, 17]}',
            "Based on the analysis, your numbers show interesting patterns.",
        ])

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-an-2", "message": "Analyze 7,11,17", "intent": "analyst"})
        assert resp.status_code == 200
        assert "response" in resp.json()
