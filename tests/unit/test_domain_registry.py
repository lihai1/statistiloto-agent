"""Unit tests for domain_registry lucky/saved numbers — no DB needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.domain_registry import get_domain_definition, detect_topic, explain


class TestLuckyNumbersDefinition:
    """Tests for lucky_numbers domain entries (#20)."""

    def test_en_definition(self):
        result = get_domain_definition("lucky_numbers", "en")
        assert result is not None
        assert "pinned" in result.lower()
        assert "every generated form" in result.lower()

    def test_he_definition(self):
        result = get_domain_definition("lucky_numbers", "he")
        assert result is not None
        assert "מספרי מזל" in result
        assert "קבועים" in result

    def test_detect_topic_en(self):
        assert detect_topic("what are lucky numbers?") == "lucky_numbers"

    def test_detect_topic_he(self):
        assert detect_topic("מה זה מספרי מזל?") == "lucky_numbers"

    def test_explain_includes_disclaimer_en(self):
        result = explain("lucky_numbers", "en")
        assert result is not None
        assert "pinned" in result.lower()
        assert "historical" in result.lower()  # disclaimer


class TestSavedNumbersDefinition:
    """Tests for saved_numbers domain entries (#20)."""

    def test_en_definition(self):
        result = get_domain_definition("saved_numbers", "en")
        assert result is not None
        assert "bookmarked" in result.lower()
        assert "wallet" in result.lower()

    def test_he_definition(self):
        result = get_domain_definition("saved_numbers", "he")
        assert result is not None
        assert "מספרים שמורים" in result
        assert "ארנק" in result

    def test_detect_topic_en(self):
        assert detect_topic("what are saved numbers?") == "saved_numbers"

    def test_detect_topic_he(self):
        assert detect_topic("מה זה מספרים שמורים?") == "saved_numbers"

    def test_explain_includes_disclaimer_he(self):
        result = explain("saved_numbers", "he")
        assert result is not None
        assert "ארנק" in result
