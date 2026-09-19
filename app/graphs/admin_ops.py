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
import os
from functools import lru_cache
from typing import Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from app.config.settings import get_tier_config
from app.llm.router import get_llm
from app.llm.limiter import llm_invoke, llm_stream_invoke
from app.metering import meter_llm
from app.prompt_builder import build_prompt, format_run_data
from app.graphs.common import format_history, append_history, make_retrieve_node, make_hitl_gate, emit_step
from app.graphs.tool_parser import parse_tool_call

log = logging.getLogger(__name__)


# ── Data-driven admin commands (loaded from YAML) ─────────────

@lru_cache(maxsize=1)
def load_admin_commands() -> list[dict]:
    """Load admin command keyword→tool mappings from YAML (cached at module level).

    The YAML file (app/rag/admin_commands.yaml) defines commands with:
      - keywords: list of strings to match in the user message
      - tool: the tool name to invoke
      - args: default arguments
      - description: human-readable description
      - dynamic_args (optional): args computed from tool_args/state at runtime

    Returns a list of command dicts. Falls back to an empty list if the
    file cannot be loaded.
    """
    import yaml
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "rag", "admin_commands.yaml")
    yaml_path = os.path.normpath(yaml_path)
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        commands = data.get("commands", [])
        for cmd in commands:
            _validate_dynamic_args(cmd.get("dynamic_args", {}), cmd.get("tool", "?"))
        log.debug("[admin_ops] Loaded %d admin commands from %s", len(commands), yaml_path)
        return commands
    except Exception as e:
        log.warning("[admin_ops] Failed to load admin_commands.yaml: %s — using empty list", e)
        return []


def _validate_dynamic_args(dynamic: dict, tool: str) -> None:
    """Fail fast on malformed dynamic_args specs (replaces the old eval())."""
    if not isinstance(dynamic, dict):
        raise ValueError(f"admin_commands[{tool}]: dynamic_args must be a mapping")
    for arg_name, spec in dynamic.items():
        if not isinstance(spec, dict) or "from" not in spec:
            raise ValueError(
                f"admin_commands[{tool}].dynamic_args.{arg_name}: "
                "expected {from: <tool_arg>, default: <value>} or {from: <tool_arg>, or: message}"
            )
        if "default" not in spec and spec.get("or") != "message":
            raise ValueError(
                f"admin_commands[{tool}].dynamic_args.{arg_name}: "
                "needs a 'default' literal or 'or: message'"
            )


def _resolve_dynamic_args(dynamic: dict, tool_args: dict, state: dict) -> dict:
    """Resolve dynamic_args specs to concrete values (no eval).

    Spec: {arg_name: {from: <tool_args key>, default: <literal>}}
       or {arg_name: {from: <tool_args key>, or: message}}  # user's raw message
    """
    resolved = {}
    for arg_name, spec in (dynamic or {}).items():
        try:
            value = tool_args.get(spec["from"])
            if value is None:
                value = state["message"].strip() if spec.get("or") == "message" else spec.get("default")
            resolved[arg_name] = value
        except Exception:
            continue  # keep YAML default if resolution fails
    return resolved


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
    emit_step("plan")
    user_sub = state["user_sub"]
    session_id = state["session_id"]
    hist_len = len(state.get("history", []))
    log.info("[admin_ops.plan] START user=%s session=%s history_len=%d", user_sub, session_id, hist_len)
    try:
        from app.llm.config_store import get_llm_with_tools
        from app.tools.registry import get_tool_definitions
        from app.graphs.tool_parser import parse_tool_call_native

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

        # Try native function calling first (bind_tools), fall back to text parsing.
        all_defs = get_tool_definitions()
        authorized_defs = [d for d in all_defs if d["name"] in cfg.allowed_tools]
        llm_with_tools = get_llm_with_tools(llm, authorized_defs)
        resp = llm_invoke(llm_with_tools, prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)

        # Try native parsing first (from AIMessage.tool_calls), fall back to text.
        planned_tool, tool_args = parse_tool_call_native(resp)
        if planned_tool is None:
            planned_tool, tool_args = parse_tool_call(content)

        # Small-model guard: for well-known admin phrases, force the right tool
        # even if the LLM picked a different one (or none).
        # The keyword→tool mappings are data-driven from admin_commands.yaml.
        message_lower = state["message"].lower()
        expected = None

        # Data-driven keyword matching from YAML.
        for cmd in load_admin_commands():
            keywords = cmd.get("keywords", [])
            if any(k in message_lower for k in keywords):
                tool_name = cmd["tool"]
                # Start with default args, then apply dynamic args.
                args = dict(cmd.get("args", {}))
                args.update(_resolve_dynamic_args(cmd.get("dynamic_args", {}), tool_args, state))
                expected = (tool_name, args)
                break

        # Fall back to app-path-based logic (not data-driven — depends on
        # extracting a path from the message and complex conditionals).
        if expected is None:
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
    emit_step("execute")
    planned_tool = state.get("planned_tool")
    tool_args = state.get("tool_args", {})
    user_sub = state["user_sub"]
    session_id = state["session_id"]

    if not planned_tool or planned_tool == "none":
        return {"tool_result": {"status": "no_action", "message": "No admin action needed."}}

    log.info("[admin_ops.execute] START user=%s session=%s tool=%s args=%s", user_sub, session_id, planned_tool, tool_args)
    from app.tool_executor import execute_tool as _execute
    try:
        # Single dispatcher: JWT re-authorization + full tool coverage.
        result = _execute(planned_tool, tool_args, state.get("jwt_token", ""))
        log.info("[admin_ops.execute] SUCCESS user=%s session=%s tool=%s", user_sub, session_id, planned_tool)
        return {"tool_result": result}
    except PermissionError as e:
        log.warning("[admin_ops.execute] DENIED user=%s session=%s tool=%s: %s", user_sub, session_id, planned_tool, e)
        return {"tool_result": {"error": str(e), "tool": planned_tool, "denied": True}}
    except Exception as e:
        log.error("[admin_ops.execute] ERROR user=%s session=%s tool=%s msg=%s", user_sub, session_id, planned_tool, e, exc_info=True)
        return {"tool_result": {"error": str(e), "tool": planned_tool}}


@meter_llm
def finalize(state: AdminOpsState) -> dict:
    """Transform the tool result into readable natural language.

    Like the analyst's finalize, this invokes the LLM to format the structured
    tool result into concise NL per the system prompt's grounding rules.
    Falls back to a plain summary if the LLM call fails.
    """
    emit_step("finalize")
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
        # .stream() so /chat/stream emits token events for this node.
        resp = llm_stream_invoke(llm, prompt)
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
