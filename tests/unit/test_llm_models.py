"""Unit tests for the /llm-models response shaping — no DB / no Ollama needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.main import _parse_ollama_tags, _GEMINI_MODELS, _OPENAI_MODELS, _ANTHROPIC_MODELS


class TestParseOllamaTags:
    def test_extracts_name_size_and_capabilities(self):
        data = {
            "models": [
                {"name": "llama3.1:8b", "size": 4661214619, "capabilities": ["completion", "tools"]},
                {"name": "nomic-embed-text", "size": 273268736, "capabilities": ["embedding"]},
                {"model": "llava:7b", "size": 4768200800, "capabilities": ["completion", "vision"]},
            ]
        }
        result = _parse_ollama_tags(data)
        assert result == [
            {"name": "llama3.1:8b", "size": 4661214619, "capabilities": ["completion", "tools"]},
            {"name": "nomic-embed-text", "size": 273268736, "capabilities": ["embedding"]},
            {"name": "llava:7b", "size": 4768200800, "capabilities": ["completion", "vision"]},
        ]

    def test_thinking_capability_preserved(self):
        data = {"models": [{"name": "deepseek-r1:8b", "size": 1, "capabilities": ["completion", "thinking"]}]}
        result = _parse_ollama_tags(data)
        assert result == [{"name": "deepseek-r1:8b", "size": 1, "capabilities": ["completion", "thinking"]}]

    def test_missing_capabilities_defaults_to_empty(self):
        """Older Ollama servers omit the capabilities field."""
        data = {"models": [{"name": "llama3.1:8b", "size": 100}]}
        result = _parse_ollama_tags(data)
        assert result == [{"name": "llama3.1:8b", "size": 100, "capabilities": []}]

    def test_missing_size_defaults_to_zero(self):
        data = {"models": [{"name": "llama3.1:8b", "capabilities": ["completion"]}]}
        result = _parse_ollama_tags(data)
        assert result == [{"name": "llama3.1:8b", "size": 0, "capabilities": ["completion"]}]

    def test_non_int_size_defaults_to_zero(self):
        data = {"models": [{"name": "x:1b", "size": "oops", "capabilities": []}]}
        result = _parse_ollama_tags(data)
        assert result == [{"name": "x:1b", "size": 0, "capabilities": []}]

    def test_non_list_capabilities_defaults_to_empty(self):
        data = {"models": [{"name": "x:1b", "size": 1, "capabilities": "oops"}]}
        result = _parse_ollama_tags(data)
        assert result == [{"name": "x:1b", "size": 1, "capabilities": []}]

    def test_skips_models_without_name(self):
        data = {"models": [{"size": 1, "capabilities": ["completion"]}, {"name": "", "size": 1, "capabilities": []}]}
        assert _parse_ollama_tags(data) == []

    def test_empty_response(self):
        assert _parse_ollama_tags({}) == []
        assert _parse_ollama_tags({"models": []}) == []

    def test_capabilities_coerced_to_strings(self):
        data = {"models": [{"name": "m:1b", "size": 1, "capabilities": ["tools", 1, True]}]}
        result = _parse_ollama_tags(data)
        assert result == [{"name": "m:1b", "size": 1, "capabilities": ["tools", "1", "True"]}]


class TestStaticProviderLists:
    """Sanity: the static lists used for non-Ollama providers are non-empty."""

    def test_gemini_models(self):
        assert len(_GEMINI_MODELS) > 0

    def test_openai_models(self):
        assert len(_OPENAI_MODELS) > 0

    def test_anthropic_models(self):
        assert len(_ANTHROPIC_MODELS) > 0
