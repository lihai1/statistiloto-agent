"""Unit tests for tool parsing — native function calling and text fallback.

Verifies that:
  - parse_tool_call_native() extracts tool calls from AIMessage.tool_calls
  - parse_tool_call() (text-based) still works as fallback
  - get_tool_definitions() returns valid LangChain tool schemas
  - get_llm_with_tools() binds tools when supported, returns plain LLM otherwise
"""

import pytest

from app.graphs.tool_parser import parse_tool_call, parse_tool_call_native
from app.tools.registry import get_tool_definitions


# ── Native tool call parsing ──────────────────────────────────


class TestParseNativeToolCall:
    def _make_ai_message(self, content: str = "", tool_calls: list | None = None):
        """Create a minimal fake AIMessage-like object."""
        class FakeAIMessage:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = tool_calls or []
        return FakeAIMessage(content, tool_calls)

    def test_parse_native_tool_call(self):
        msg = self._make_ai_message(
            tool_calls=[{"name": "get_statistics", "args": {"group_size": 2, "strength": "hot"}}],
        )
        tool, args = parse_tool_call_native(msg)
        assert tool == "get_statistics"
        assert args == {"group_size": 2, "strength": "hot"}

    def test_parse_native_no_tool_calls(self):
        msg = self._make_ai_message(content="I don't need a tool for this.")
        tool, args = parse_tool_call_native(msg)
        assert tool is None
        assert args == {}

    def test_parse_native_multiple_takes_first(self):
        msg = self._make_ai_message(
            tool_calls=[
                {"name": "get_statistics", "args": {"group_size": 2}},
                {"name": "generate_form", "args": {"how_many": 5}},
            ],
        )
        tool, args = parse_tool_call_native(msg)
        assert tool == "get_statistics"

    def test_parse_native_empty_args(self):
        msg = self._make_ai_message(
            tool_calls=[{"name": "trigger_scraper", "args": {}}],
        )
        tool, args = parse_tool_call_native(msg)
        assert tool == "trigger_scraper"
        assert args == {}

    def test_parse_native_none_attribute(self):
        """AIMessage with tool_calls=None should not crash."""
        class FakeMsg:
            content = "no tools"
            tool_calls = None
        tool, args = parse_tool_call_native(FakeMsg())
        assert tool is None
        assert args == {}


# ── Text-based fallback parsing (regression) ──────────────────


class TestTextFallback:
    def test_text_fallback_still_works(self):
        tool, args = parse_tool_call('TOOL: get_statistics ARGS: {"group_size": 2}')
        assert tool == "get_statistics"
        assert args == {"group_size": 2}

    def test_text_fallback_malformed_json(self):
        tool, args = parse_tool_call('TOOL: get_statistics ARGS: {broken')
        assert tool == "get_statistics"
        assert args == {}

    def test_text_fallback_no_tool_line(self):
        tool, args = parse_tool_call("Just a regular response, no tool call.")
        assert tool is None
        assert args == {}

    def test_text_fallback_none(self):
        tool, args = parse_tool_call("TOOL: none")
        assert tool == "none"


# ── Tool definitions ──────────────────────────────────────────


class TestToolDefinitions:
    def test_tool_definitions_schema(self):
        defs = get_tool_definitions()
        assert isinstance(defs, list)
        assert len(defs) > 0
        names = {d["name"] for d in defs}
        # Core tools should be present.
        assert "get_statistics" in names
        assert "generate_form" in names
        assert "analyze" in names
        assert "save_numbers" in names

    def test_tool_definitions_have_required_fields(self):
        defs = get_tool_definitions()
        for d in defs:
            assert "name" in d, f"Tool def missing 'name': {d}"
            assert "description" in d, f"Tool def missing 'description': {d}"
            assert "parameters" in d, f"Tool def missing 'parameters': {d}"
            # parameters should be a JSON schema dict with at least 'type'.
            assert isinstance(d["parameters"], dict)

    def test_tool_definitions_descriptions_are_meaningful(self):
        defs = get_tool_definitions()
        for d in defs:
            assert len(d["description"]) > 10, f"Tool '{d['name']}' has trivial description"


# ── get_llm_with_tools ────────────────────────────────────────


class TestGetLlmWithTools:
    def test_get_llm_with_tools_supported(self):
        """LLM with bind_tools method should return bound LLM."""
        class FakeLLMWithTools:
            def bind_tools(self, tools):
                self._bound = tools
                return self
        from app.llm.config_store import get_llm_with_tools
        llm = FakeLLMWithTools()
        tool_defs = [{"name": "test", "description": "test", "parameters": {"type": "object"}}]
        result = get_llm_with_tools(llm, tool_defs)
        assert result is llm
        assert result._bound == tool_defs

    def test_get_llm_with_tools_unsupported(self):
        """LLM without bind_tools should return plain LLM (no error)."""
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from app.llm.config_store import get_llm_with_tools
        llm = FakeListChatModel(responses=["test"])
        tool_defs = [{"name": "test", "description": "test", "parameters": {"type": "object"}}]
        result = get_llm_with_tools(llm, tool_defs)
        assert result is llm  # unchanged — no binding

    def test_get_llm_with_tools_bind_raises(self):
        """If bind_tools raises, should fall back to plain LLM."""
        class FakeLLMBrokenBind:
            def bind_tools(self, tools):
                raise TypeError("This model does not support tool calling")
        from app.llm.config_store import get_llm_with_tools
        llm = FakeLLMBrokenBind()
        tool_defs = [{"name": "test", "description": "test", "parameters": {"type": "object"}}]
        result = get_llm_with_tools(llm, tool_defs)
        assert result is llm  # fell back to plain LLM
