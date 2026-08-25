"""Eval: Grounding tests.

Verifies that:
  - Tool failure does not produce fabricated statistics (fail-closed).
  - Historical frequency is not described as improved winning probability.
  - Raw tool JSON is not dumped to the user.
"""
import pytest

pytestmark = pytest.mark.integration


class TestGrounding:
    """Grounding and fail-closed behavior."""

    def test_tool_failure_no_fabrication(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """When the statistics tool returns an error, the LLM should not fabricate data."""
        from app.tools import lottery_grpc
        lottery_grpc._mock_client["get_statistics"] = lambda **kw: {
            "groups": [], "error": "Lottery service unavailable"
        }

        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 5, "group_size": 2, "strength": "hot"}',
            "I couldn't retrieve the statistics right now. The lottery service is unavailable.",
        ])

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-gr-1", "message": "What are the hot pairs?", "intent": "analyst"})
        assert resp.status_code == 200
        response_text = resp.json().get("response", "")
        # Should NOT contain fabricated statistics.
        assert "17 + 31" not in response_text
        assert "appeared 9 times" not in response_text
        # Should indicate failure.
        assert any(w in response_text.lower() for w in ["couldn't", "unavailable", "unable", "could not"])

    def test_no_raw_json_in_response(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """The LLM should not dump raw JSON to the user."""
        from app.tools import lottery_grpc
        lottery_grpc._mock_client["get_statistics"] = lambda **kw: {
            "groups": [{"numbers": [17, 31], "count": 9}, {"numbers": [7, 24], "count": 8}]
        }

        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 5, "group_size": 2, "strength": "hot"}',
            "The most frequent pairs are:\n1. 17 + 31 — 9 appearances\n2. 7 + 24 — 8 appearances",
        ])

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-gr-2", "message": "Top 5 hot pairs", "intent": "analyst"})
        assert resp.status_code == 200
        response_text = resp.json().get("response", "")
        assert '"groups"' not in response_text
        assert '"numbers"' not in response_text
        assert '{"' not in response_text

    def test_frequency_not_described_as_probability(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Historical frequency should not be described as improved winning probability."""
        from app.tools import lottery_grpc
        lottery_grpc._mock_client["get_statistics"] = lambda **kw: {
            "groups": [{"numbers": [17, 31], "count": 9}]
        }

        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 5, "group_size": 2, "strength": "hot"}',
            "The pair 17 + 31 appeared 9 times. "
            "These are historical observations only and do not imply a higher probability in the next draw.",
        ])

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-gr-3", "message": "What are the hot pairs?", "intent": "analyst"})
        assert resp.status_code == 200
        response_text = resp.json().get("response", "")
        assert "higher chance" not in response_text.lower()
        assert "more likely to win" not in response_text.lower()
        assert "due to appear" not in response_text.lower()
        disclaimer_markers = ["do not imply", "historical observations only", "אלה נתוני עבר", "לא מעידים"]
        assert any(m in response_text for m in disclaimer_markers), \
            f"Response missing disclaimer: {response_text}"
