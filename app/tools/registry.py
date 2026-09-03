"""Tool registry — classifies tools as read or write for HITL gating.

Also provides LangChain-compatible tool definitions for native function calling
via ChatOllama.bind_tools().

HITL (human-in-the-loop) approval is required before the agent executes
ANY tool that writes, updates, or deletes data. Read-only tools execute
without HITL.
"""

from __future__ import annotations

# Tools that write/update/delete data → always require HITL approval before executing.
WRITE_TOOLS = frozenset({
    "save_numbers",       # POST to Java BFF → writes saved numbers to DB
    "trigger_scraper",    # triggers Go scraper → writes lottery draw data
    "edit_file",          # edit agent source code (admin only)
})

# Read-only tools → execute without HITL.
READ_TOOLS = frozenset({
    "generate_form",      # Go gRPC → computes forms (no DB write)
    "get_statistics",     # Go gRPC → reads statistics
    "analyze",            # Go gRPC → reads historical matches
    "list_saved_numbers", # Java BFF GET → reads saved numbers
    "query_audit_log",    # DB SELECT → reads audit log
    "read_token_usage",   # DB SELECT → reads token stats
    "search_web",         # public web search (admin only)
    "read_code",          # read agent source code (admin only)
    "list_files",         # list agent source files (admin only)
    "list_db_tables",     # DB SELECT → list tables in a schema (admin only)
    "query_db",           # DB SELECT → read-only SQL query (admin only)
})

# Union of all known tools.
ALL_TOOLS = WRITE_TOOLS | READ_TOOLS


def is_write_tool(tool_name: str) -> bool:
    """Check if a tool writes/updates/deletes data (requires HITL)."""
    return tool_name in WRITE_TOOLS


def is_read_tool(tool_name: str) -> bool:
    """Check if a tool is read-only (no HITL needed)."""
    return tool_name in READ_TOOLS


# ── Tool definitions for native function calling ──────────────

_TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "get_statistics",
        "description": "Get the most or least frequent number groups from historical lottery draws. Use for 'hot/cold pairs/triples/singles' requests.",
        "parameters": {
            "type": "object",
            "properties": {
                "how_many": {"type": "integer", "description": "Number of top groups to return (default 10)."},
                "group_size": {"type": "integer", "description": "Group size 1-6 (1=singles, 2=pairs, 3=triples, etc.)."},
                "strength": {"type": "string", "enum": ["hot", "cold"], "description": "Hot = most frequent, cold = least frequent."},
            },
            "required": ["group_size", "strength"],
        },
    },
    {
        "name": "generate_form",
        "description": "Generate lottery forms (combinations of numbers) based on historical frequency. Use for 'generate forms' or 'create combinations' requests.",
        "parameters": {
            "type": "object",
            "properties": {
                "how_many": {"type": "integer", "description": "Number of forms to generate."},
                "form_type": {"type": "integer", "description": "Form size (default 6 = regular lotto)."},
                "strength": {"type": "integer", "description": "2=STRONG/hot (default), 1=WEAK/cold."},
                "will_be": {"type": "array", "items": {"type": "integer"}, "description": "Lucky numbers to include in every form."},
            },
            "required": ["how_many"],
        },
    },
    {
        "name": "analyze",
        "description": "Analyze user-selected numbers against historical draws. Returns frequency for all subset sizes 1-6 of the selected numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "form": {"type": "array", "items": {"type": "integer"}, "description": "The user's selected numbers (1-6 regular numbers)."},
            },
            "required": ["form"],
        },
    },
    {
        "name": "save_numbers",
        "description": "Save a set of lottery numbers for the user. Requires human approval before executing.",
        "parameters": {
            "type": "object",
            "properties": {
                "numbers": {"type": "array", "items": {"type": "integer"}, "description": "The numbers to save (1-99)."},
                "category": {"type": "string", "description": "Optional category label (default 'default')."},
            },
            "required": ["numbers"],
        },
    },
    {
        "name": "list_saved_numbers",
        "description": "List the user's previously saved lottery numbers.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "trigger_scraper",
        "description": "Trigger the lottery draw scraper to fetch new draw data. Requires human approval. Admin only.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "query_audit_log",
        "description": "Query the audit log for recent admin actions. Admin only.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max entries to return (default 50)."},
            },
        },
    },
    {
        "name": "read_token_usage",
        "description": "Read token usage statistics for LLM calls. Admin only.",
        "parameters": {
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Number of days to look back (default 7)."},
            },
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information. Admin only.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "limit": {"type": "integer", "description": "Max results (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_code",
        "description": "Read a file from the agent's source tree. Admin only.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file (e.g. 'app/main.py')."},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_files",
        "description": "List files under a directory in the agent's source tree. Admin only.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path (e.g. 'app/graphs')."},
            },
            "required": ["directory"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file in the agent's source tree. Requires human approval. Admin only.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file to edit."},
                "old_string": {"type": "string", "description": "Text to find and replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "content": {"type": "string", "description": "Full new file content (alternative to old/new_string)."},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_db_tables",
        "description": "List tables and columns in a database schema. Admin only.",
        "parameters": {
            "type": "object",
            "properties": {
                "schema": {"type": "string", "description": "Schema name (default 'agent')."},
            },
        },
    },
    {
        "name": "query_db",
        "description": "Run a read-only SQL SELECT query against any schema. Admin only.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The SQL SELECT query to run."},
                "limit": {"type": "integer", "description": "Max rows to return (default 50)."},
            },
            "required": ["sql"],
        },
    },
]


def get_tool_definitions() -> list[dict]:
    """Return LangChain-compatible tool definitions for native function calling.

    Each definition is a dict with 'name', 'description', and 'parameters' (JSON schema).
    Used by ChatOllama.bind_tools() when the model supports native tool calling.
    """
    return [dict(d) for d in _TOOL_DEFINITIONS]
