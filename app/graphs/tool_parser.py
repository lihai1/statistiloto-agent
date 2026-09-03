"""Parse LLM tool-call responses — native function calling + text fallback.

Shared by analyst and admin_ops worker subgraphs.

Two parsing modes:
  1. Native: extract tool calls from AIMessage.tool_calls (Ollama bind_tools).
  2. Text fallback: parse 'TOOL: <name> ARGS: <json>' from LLM text output.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Tuple

log = logging.getLogger(__name__)


def parse_tool_call_native(message: Any) -> Tuple[Optional[str], dict]:
    """Extract the first tool call from an AIMessage's tool_calls attribute.

    Returns (tool_name, args_dict). tool_name is None if no tool calls.
    Falls back to text parsing if the message has content with a TOOL: line
    but no structured tool_calls.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        first = tool_calls[0]
        name = first.get("name")
        args = first.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        return name, args
    # No structured tool calls — try text fallback if content has TOOL:.
    content = getattr(message, "content", "")
    if content and isinstance(content, str):
        return parse_tool_call(content)
    return None, {}


def parse_tool_call(content: str) -> Tuple[Optional[str], dict]:
    """Parse a 'TOOL: <tool_name> ARGS: <json_args>' line from LLM output.

    Returns (tool_name, args_dict). tool_name is None if no TOOL: line found.
    tool_name is the string "none" if the LLM explicitly declined.
    args_dict is {} if no args or args fail to parse as JSON.

    The format is case-insensitive on the keywords and tolerates optional
    whitespace before ARGS:.
    """
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("TOOL:"):
            rest = stripped[5:].strip()  # everything after "TOOL:"
            parts = re.split(r"(?i)\s*ARGS:\s*", rest, maxsplit=1)
            tool_name = parts[0].strip()
            args_str = parts[1].strip() if len(parts) > 1 else ""
            args: dict = {}
            if args_str:
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, ValueError):
                    args = {}
            return tool_name, args
    return None, {}
