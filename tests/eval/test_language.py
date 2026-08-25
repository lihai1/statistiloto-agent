"""Eval: Language matching tests.

Verifies that:
  - Hebrew question → Hebrew response.
  - English question → English response.
  - Number combinations are formatted readably.
"""
import pytest
import re

pytestmark = pytest.mark.integration

HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


def _is_hebrew(text: str) -> bool:
    return bool(HEBREW_RE.search(text))


def _has_english_words(text: str) -> bool:
    english_markers = ["the ", "and ", "are ", "is ", "you ", "your ", "this ", "that "]
    lower = text.lower()
    return any(m in lower for m in english_markers)


class TestLanguageMatching:
    """Language matching behavior."""

    def test_hebrew_question_hebrew_response(self, client, free_headers, set_llm_responses):
        """Hebrew question should receive a Hebrew response."""
        set_llm_responses([
            "מספרים חמים הם מספרים שהופיעו בתדירות גבוהה יותר בעבר. "
            "אלה נתוני עבר בלבד ואינם מעידים על סיכוי גבוה יותר בהגרלה הבאה."
        ])

        resp = client.post("/chat", headers=free_headers,
            json={"session_id": "eval-lang-1", "message": "מה זה מספרים חמים?"})
        assert resp.status_code == 200
        response_text = resp.json().get("response", "")
        assert _is_hebrew(response_text), f"Expected Hebrew response, got: {response_text}"

    def test_english_question_english_response(self, client, free_headers, set_llm_responses):
        """English question should receive an English response."""
        set_llm_responses([
            "Hot numbers are numbers that have appeared more frequently in the past. "
            "These are historical observations only and do not imply a higher probability in the next draw."
        ])

        resp = client.post("/chat", headers=free_headers,
            json={"session_id": "eval-lang-2", "message": "What are hot numbers?"})
        assert resp.status_code == 200
        response_text = resp.json().get("response", "")
        assert _has_english_words(response_text), f"Expected English response, got: {response_text}"
        assert not _is_hebrew(response_text), f"Expected English, got Hebrew: {response_text}"

    def test_number_formatting_readable(self, client, paid_headers, mock_tool_clients, set_llm_responses):
        """Number combinations should be formatted readably, not as raw arrays."""
        from app.tools import lottery_grpc
        lottery_grpc._mock_client["get_statistics"] = lambda **kw: {
            "groups": [{"numbers": [17, 31], "count": 9}]
        }

        set_llm_responses([
            'TOOL: get_statistics ARGS: {"how_many": 5, "group_size": 2, "strength": "hot"}',
            "The most frequent pair is 17 + 31 with 9 appearances.",
        ])

        resp = client.post("/chat", headers=paid_headers,
            json={"session_id": "eval-lang-3", "message": "Top hot pairs", "intent": "analyst"})
        assert resp.status_code == 200
        response_text = resp.json().get("response", "")
        assert "17 + 31" in response_text or "17, 31" in response_text
        assert "[17, 31]" not in response_text
