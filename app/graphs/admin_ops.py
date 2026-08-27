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
from app.prompt_builder import build_prompt, format_run_data
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
    lang: Optional[str]
    planned_tool: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[dict]
    response: Optional[str]
    draft: Optional[str]


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
        cfg = get_tier_config(state["tier"])
        tools_str = ", ".join(cfg.allowed_tools)
        hist = format_history(state.get("history", []))
        lang = state.get("lang") or "en"

        # Phase 5: use compact planner prompt with admin-specific examples.
        prompt = build_prompt(
            route="ambiguous_planner",
            language=lang,
            user_message=state["message"],
            authorized_tools=tools_str,
            history=hist,
        )
        # Append admin-specific tool examples inline (compact).
        prompt += (
            "\n\nADMIN TOOLS:\n"
            "  'token usage'/'costs' → read_token_usage {\"days\": 7}\n"
            "  'audit'/'logs' → query_audit_log {\"limit\": 50}\n"
            "  'scraper'/'refresh draws' → trigger_scraper {}\n"
            "  'search the web' → search_web {\"query\": \"...\", \"limit\": 5}\n"
            "  'show me'/'read' a .py file → read_code {\"file_path\": \"app/...\"}\n"
            "  'list files' → list_files {\"directory\": \"app/...\"}\n"
            "  'edit'/'replace' a file → edit_file {\"file_path\": \"...\", \"old_string\": \"...\", \"new_string\": \"...\"}\n"
            "  'tables in'/'schema' → list_db_tables {\"schema\": \"agent\"}\n"
            "  'users'/'sessions' → query_db {\"sql\": \"SELECT ...\", \"limit\": 50}\n"
            "  No match → TOOL: none"
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
        elif any(k in message_lower for k in ("tables in", "schema", "what tables")):
            expected = ("list_db_tables", {"schema": tool_args.get("schema", "agent")})
        elif any(k in message_lower for k in ("show me the users", "who is using", "list users", "show users", "all users")):
            expected = ("query_db", {
                "sql": "SELECT DISTINCT user_sub, tier FROM agent.token_usage ORDER BY tier",
                "limit": tool_args.get("limit", 50),
            })
        elif any(k in message_lower for k in ("chat sessions", "show sessions", "list sessions", "active sessions")):
            expected = ("query_db", {
                "sql": "SELECT user_sub, session_id, title, message_count, updated_at FROM agent.chat_sessions ORDER BY updated_at DESC",
                "limit": tool_args.get("limit", 50),
            })
        else:
            app_path = _extract_app_path(state["message"])
            if app_path:
                if any(k in message_lower for k in ("show me", "read", "view")) and app_path.endswith(".py"):
                    expected = ("read_code", {"file_path": tool_args.get("file_path", app_path)})
                elif any(k in message_lower for k in ("list files", "files in")):
                    expected = ("list_files", {"directory": tool_args.get("directory", app_path)})
                elif any(k in message_lower for k in ("edit", "replace", "change")) and app_path.endswith(".py"):
                    content_arg = tool_args.get("content")
                    old_string = tool_args.get("old_string")
                    new_string = tool_args.get("new_string")
                    quoted = _extract_quoted_content(state["message"])
                    if content_arg is None and ("entire content" in message_lower or "full content" in message_lower or "replace all" in message_lower) and quoted:
                        content_arg = quoted
                    elif content_arg is None and (old_string is None or new_string is None):
                        content_arg = quoted
                    expected = ("edit_file", {
                        "file_path": tool_args.get("file_path", app_path),
                        "old_string": old_string,
                        "new_string": new_string,
                        "content": content_arg,
                    })
        if expected:
            planned_tool, tool_args = expected
            log.info("[admin_ops.plan] GUARD user=%s session=%s tool=%s", user_sub, session_id, planned_tool)

        log.info("[admin_ops.plan] SUCCESS user=%s session=%s planned_tool=%s", user_sub, session_id, planned_tool)
        return {
            "draft": content,
            "planned_tool": planned_tool,
            "tool_args": tool_args,
            "tool_result": {"planned_action": content},
            "_usage": getattr(resp, "usage_metadata", None),
            "_response_metadata": getattr(resp, "response_metadata", None),
        }
    except Exception as e:
        log.error("[admin_ops.plan] ERROR user=%s session=%s msg=%s", user_sub, session_id, e, exc_info=True)
        raise


maybe_hitl = make_hitl_gate("admin_ops", execute_node="execute", finalize_node="finalize")


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
        elif planned_tool == "list_db_tables":
            result = admin_ops.list_db_tables(
                claims, schema=tool_args.get("schema", "agent"),
            )
        elif planned_tool == "query_db":
            result = admin_ops.query_db(
                claims,
                sql=tool_args.get("sql", ""),
                limit=tool_args.get("limit", 50),
            )
        else:
            result = {"status": "unknown_action", "message": f"Unknown tool: {planned_tool}"}

        log.info("[admin_ops.execute] SUCCESS user=%s session=%s tool=%s", user_sub, session_id, planned_tool)
        return {"tool_result": result}
    except Exception as e:
        log.error("[admin_ops.execute] ERROR user=%s session=%s tool=%s msg=%s", user_sub, session_id, planned_tool, e, exc_info=True)
        raise


@meter_llm
def finalize(state: AdminOpsState) -> dict:
    """Transform the tool result into readable natural language.

    Like the analyst's finalize, this invokes the LLM to format the structured
    tool result into concise NL per the system prompt's grounding rules.
    Falls back to a plain summary if the LLM call fails.
    """
    user_sub = state["user_sub"]
    session_id = state["session_id"]
    tool_result = state.get("tool_result")
    planned_tool = state.get("planned_tool")
    lang = state.get("lang") or "en"

    # No tool was planned — use the draft or a simple fallback.
    if not planned_tool or planned_tool == "none":
        draft = state.get("draft", "")
        if draft and not draft.strip().upper().startswith("TOOL:"):
            response = draft
        else:
            response = "No admin action needed."
        updated_history = append_history(state, response)
        return {"response": response, "history": updated_history}

    # A tool was executed — format the result into NL.
    if not tool_result:
        response = "Operation completed."
        updated_history = append_history(state, response)
        return {"response": response, "history": updated_history}

    try:
        llm = get_llm()
        hist = format_history(state.get("history", []))
        run_data = format_run_data(planned_tool or "", tool_result)
        prompt = build_prompt(
            route="admin_finalizer",
            language=lang,
            user_message=state["message"],
            run_data=run_data,
            history=hist,
        )
        resp = llm.invoke(prompt)
        response = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as e:
        log.error("[admin_ops.finalize] LLM formatting failed: %s — using summary", e)
        # Fallback: brief summary instead of raw JSON
        if isinstance(tool_result, dict):
            if "entries" in tool_result:
                response = f"Found {len(tool_result['entries'])} entries."
            elif "status" in tool_result:
                response = f"Operation status: {tool_result['status']}."
            else:
                response = f"Operation completed. {list(tool_result.keys())}"
        else:
            response = "Operation completed."

    updated_history = append_history(state, response)
    log.info("[admin_ops.finalize] SUCCESS user=%s session=%s response_len=%d", user_sub, session_id, len(response))
    return {"response": response, "history": updated_history}


def build_admin_ops_graph():
    """Build the admin ops worker subgraph."""
    g = StateGraph(AdminOpsState)
    g.add_node("retrieve", retrieve_docs)
    g.add_node("plan", plan_action)
    g.add_node("hitl_gate", maybe_hitl)
    g.add_node("execute", execute)
    g.add_node("finalize", finalize)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "plan")
    g.add_edge("plan", "hitl_gate")
    g.add_edge("execute", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
