"""Tool registry — classifies tools as read or write for HITL gating.

HITL (human-in-the-loop) approval is required before the agent executes
ANY tool that writes, updates, or deletes data. Read-only tools execute
without HITL.
"""

from __future__ import annotations

# Tools that write/update/delete data → always require HITL approval before executing.
WRITE_TOOLS = frozenset({
    "save_numbers",       # POST to Java BFF → writes saved numbers to DB
    "trigger_scraper",    # triggers Go scraper → writes lottery draw data
})

# Read-only tools → execute without HITL.
READ_TOOLS = frozenset({
    "generate_form",      # Go gRPC → computes forms (no DB write)
    "get_statistics",     # Go gRPC → reads statistics
    "analyze",            # Go gRPC → reads historical matches
    "list_saved_numbers", # Java BFF GET → reads saved numbers
    "query_audit_log",    # DB SELECT → reads audit log
    "read_token_usage",   # DB SELECT → reads token stats
})

# Union of all known tools.
ALL_TOOLS = WRITE_TOOLS | READ_TOOLS


def is_write_tool(tool_name: str) -> bool:
    """Check if a tool writes/updates/deletes data (requires HITL)."""
    return tool_name in WRITE_TOOLS


def is_read_tool(tool_name: str) -> bool:
    """Check if a tool is read-only (no HITL needed)."""
    return tool_name in READ_TOOLS
