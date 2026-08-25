"""Parse LLM tool-call responses in the 'TOOL: <name> ARGS: <json>' format.

Shared by analyst and admin_ops worker subgraphs.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple


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
