"""Admin Ops worker subgraph.

Operations tools (scraper trigger, audit log query, token usage read)
with HITL on any write tool call.

Admin is the owner/developer super-user. The LLM plans an action; if
the planned action involves a write tool (trigger_scraper, save_numbers),
the graph pauses for human approval before executing. Read-only actions
(query_audit_log, read_token_usage) proceed without HITL.
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from app.config.settings import get_tier_config
from app.llm.router import get_llm
from app.metering import meter_llm
from app.prompts import SYSTEM_PROMPT_WITH_TOOLS
from app.security import TokenClaims
from app.tools import admin_ops, online_search, code_editor
from app.graphs.common import format_history, append_history, make_retrieve_node, make_hitl_gate
from app.graphs.tool_parser import parse_tool_call

log = logging.getLogger(__name__)


class AdminOpsState(TypedDict):
    user_sub: str
    tier: str
    session_id: str
    message: str
    jwt_token: str
    chunks: list
    history: list
    planned_tool: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[dict]
    response: Optional[str]


def _extract_app_path(text: str) -> str | None:
    """Crudely extract an app/... path (file or directory) from the user message."""
    import re
    m = re.search(r"\b(app/[\w./-]+)\b", text)
    return m.group(1) if m else None


def _extract_quoted_content(text: str) -> str | None:
    """Extract the content in the last '...' or \"...\" of a 'replace with ...' request."""
    import re
    for pattern in (r"with '([^']+)'", r'with "([^"]+)"'):
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


retrieve_docs = make_retrieve_node("admin_ops")


@meter_llm
def plan_action(state: AdminOpsState) -> dict:
    """Use the LLM to determine what admin action to take, with conversation history.

    The LLM responds with:
      TOOL: <tool_name> ARGS: <json_args>
    """
    user_sub = state["user_sub"]
    session_id = state["session_id"]
    hist_len = len(state.get("history", []))
    log.info("[admin_ops.plan] START user=%s session=%s history_len=%d", user_sub, session_id, hist_len)
    try:
        llm = get_llm()
        ctx = "\n".join(c["text"] for c in state.get("chunks", []))
        cfg = get_tier_config(state["tier"])
        tools_str = ", ".join(cfg.allowed_tools)
        hist = format_history(state.get("history", []))
        prompt = (
            f"{SYSTEM_PROMPT_WITH_TOOLS}\n\n"
            f"You are the owner admin. Map the request to one of these tools: {tools_str}.\n"
            f"Output EXACTLY ONE LINE and nothing else:\n"
            f"  TOOL: <tool_name> ARGS: <json_args>\n\n"
            f"Worked examples (the user may ask exactly like this):\n"
            f"  Admin request: Show me the recent token usage for all users.\n"
            f"  Output: TOOL: read_token_usage ARGS: {{\"days\": 7}}\n\n"
            f"  Admin request: Show me the latest audit logs.\n"
            f"  Output: TOOL: query_audit_log ARGS: {{\"limit\": 50}}\n\n"
            f"  Admin request: Run the scraper.\n"
            f"  Output: TOOL: trigger_scraper ARGS: {{}}\n\n"
            f"  Admin request: Search the web for recent lottery regulation changes.\n"
            f"  Output: TOOL: search_web ARGS: {{\"query\": \"lottery regulation changes 2025\", \"limit\": 5}}\n\n"
            f"  Admin request: Show me app/graphs/admin_ops.py.\n"
            f"  Output: TOOL: read_code ARGS: {{\"file_path\": \"app/graphs/admin_ops.py\"}}\n\n"
            f"  Admin request: List files in app/tools.\n"
            f"  Output: TOOL: list_files ARGS: {{\"directory\": \"app/tools\"}}\n\n"
            f"  Admin request: Replace the welcome text in app/main.py.\n"
            f"  Output: TOOL: edit_file ARGS: {{\"file_path\": \"app/main.py\", \"old_string\": \"hello\", \"new_string\": \"hi\"}}\n\n"
            f"Rules:\n"
            f"  - 'token usage' or 'costs' or 'billing' -> read_token_usage\n"
            f"  - 'audit' or 'logs' -> query_audit_log\n"
            f"  - 'scraper' or 'scrape' or 'refresh draws' -> trigger_scraper\n"
            f"  - 'search the web' or 'look up' or 'find online' -> search_web\n"
            f"  - 'show me' or 'read' or 'view' a file -> read_code\n"
            f"  - 'list files' or 'files in' -> list_files\n"
            f"  - 'edit' or 'replace' or 'change' a file -> edit_file\n\n"
            f"If the request does not match any tool, output: TOOL: none"
        )
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)

        planned_tool, tool_args = parse_tool_call(content)

        # Small-model guard: for well-known admin phrases, force the right tool
        # even if the LLM picked a different one (or none).
        message_lower = state["message"].lower()
        expected = None
        if any(k in message_lower for k in ("token", "usage", "cost")):
            expected = ("read_token_usage", {"days": 7})
        elif any(k in message_lower for k in ("audit", "log")):
            expected = ("query_audit_log", {"limit": 50})
        elif any(k in message_lower for k in ("scrape", "scraper", "refresh draws")):
            expected = ("trigger_scraper", {})
        elif any(k in message_lower for k in ("search the web", "look up", "find online")):
            expected = ("search_web", {
                "query": tool_args.get("query", state["message"].strip()),
                "limit": tool_args.get("limit", 5),
            })
        else:
            app_path = _extract_app_path(state["message"])
            if app_path:
                if any(k in message_lower for k in ("show me", "read", "view")) and app_path.endswith(".py"):
                    expected = ("read_code", {"file_path": tool_args.get("file_path", app_path)})
                elif any(k in message_lower for k in ("list files", "files in")):
                    expected = ("list_files", {"directory": tool_args.get("directory", app_path)})
                elif any(k in message_lower for k in ("edit", "replace", "change")) and app_path.endswith(".py"):
                    content = tool_args.get("content")
                    old_string = tool_args.get("old_string")
                    new_string = tool_args.get("new_string")
                    quoted = _extract_quoted_content(state["message"])
                    # Full overwrite requests: if user asked to replace "entire content" and quoted the new text, prefer `content`.
                    if content is None and ("entire content" in message_lower or "full content" in message_lower or "replace all" in message_lower) and quoted:
                        content = quoted
                    elif content is None and (old_string is None or new_string is None):
                        # No args yet; use any quoted text as the full content for a simple overwrite.
                        content = quoted
                    expected = ("edit_file", {
                        "file_path": tool_args.get("file_path", app_path),
                        "old_string": old_string,
                        "new_string": new_string,
                        "content": content,
                    })
        if expected:
            planned_tool, tool_args = expected
            log.info("[admin_ops.plan] GUARD user=%s session=%s tool=%s", user_sub, session_id, planned_tool)

        log.info("[admin_ops.plan] SUCCESS user=%s session=%s planned_tool=%s", user_sub, session_id, planned_tool)
        return {
            "planned_tool": planned_tool,
            "tool_args": tool_args,
            "tool_result": {"planned_action": content},
            "_usage": getattr(resp, "usage_metadata", None),
            "_response_metadata": getattr(resp, "response_metadata", None),
        }
    except Exception as e:
        log.error("[admin_ops.plan] ERROR user=%s session=%s msg=%s", user_sub, session_id, e, exc_info=True)
        raise


maybe_hitl = make_hitl_gate("admin_ops", execute_node="execute")


def execute(state: AdminOpsState) -> dict:
    """Execute the planned admin action and append the exchange to conversation history."""
    planned_tool = state.get("planned_tool")
    tool_args = state.get("tool_args", {})
    user_sub = state["user_sub"]
    session_id = state["session_id"]
    claims = TokenClaims(
        sub=user_sub,
        tier=state["tier"],
        roles=[],
        raw_token=state.get("jwt_token", ""),
    )

    log.info("[admin_ops.execute] START user=%s session=%s tool=%s args=%s", user_sub, session_id, planned_tool, tool_args)
    try:
        if not planned_tool or planned_tool == "none":
            result = {"status": "no_action", "message": "No admin action needed."}
        elif planned_tool == "trigger_scraper":
            result = admin_ops.trigger_scraper(claims)
        elif planned_tool == "query_audit_log":
            result = admin_ops.query_audit_log(claims, limit=tool_args.get("limit", 50))
        elif planned_tool == "read_token_usage":
            result = admin_ops.read_token_usage(claims, days=tool_args.get("days", 7))
        elif planned_tool == "save_numbers":
            from app.tools.saved_numbers import save_numbers
            result = save_numbers(
                user_sub=user_sub,
                jwt_token=state.get("jwt_token", ""),
                category=tool_args.get("category", "default"),
                numbers=tool_args.get("numbers", []),
                will_be=tool_args.get("will_be"),
            )
        elif planned_tool == "search_web":
            result = online_search.search_web(
                query=tool_args.get("query", ""),
                limit=tool_args.get("limit", 5),
            )
        elif planned_tool == "read_code":
            result = code_editor.read_code(file_path=tool_args.get("file_path", ""))
        elif planned_tool == "list_files":
            result = code_editor.list_files(directory=tool_args.get("directory"))
        elif planned_tool == "edit_file":
            result = code_editor.edit_file(
                file_path=tool_args.get("file_path", ""),
                old_string=tool_args.get("old_string"),
                new_string=tool_args.get("new_string"),
                content=tool_args.get("content"),
            )
        else:
            result = {"status": "unknown_action", "message": f"Unknown tool: {planned_tool}"}

        # Serialize as JSON (not Python repr) so the Angular UI can JSON.parse it.
        import json
        response_str = json.dumps(result, default=str)

        updated_history = append_history(state, response_str)
        log.info("[admin_ops.execute] SUCCESS user=%s session=%s tool=%s", user_sub, session_id, planned_tool)
        return {"tool_result": result, "response": response_str, "history": updated_history}
    except Exception as e:
        log.error("[admin_ops.execute] ERROR user=%s session=%s tool=%s msg=%s", user_sub, session_id, planned_tool, e, exc_info=True)
        raise


def build_admin_ops_graph():
    """Build the admin ops worker subgraph."""
    g = StateGraph(AdminOpsState)
    g.add_node("retrieve", retrieve_docs)
    g.add_node("plan", plan_action)
    g.add_node("hitl_gate", maybe_hitl)
    g.add_node("execute", execute)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "hitl_gate")
    g.add_edge("execute", END)
    return g.compile()
