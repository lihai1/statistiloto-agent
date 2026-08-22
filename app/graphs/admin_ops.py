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
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

from app.config.settings import get_tier_config
from app.llm.router import get_llm
from app.metering import meter_llm
from app.rag.retriever import retrieve
from app.security import TokenClaims
from app.tools import admin_ops
from app.tools.registry import is_write_tool

log = logging.getLogger(__name__)


class AdminOpsState(TypedDict):
    user_sub: str
    tier: str
    session_id: str
    message: str
    jwt_token: str
    chunks: list
    planned_tool: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[dict]
    response: Optional[str]


def retrieve_docs(state: AdminOpsState) -> dict:
    """Retrieve from admin corpora (docs + history + user_data + ops_logs).

    Admin sees ALL users' user_data — the user_sub filter is bypassed
    for admin in the retriever.
    """
    cfg = get_tier_config(state["tier"])
    chunks = retrieve(
        query=state["message"],
        corpora=cfg.rag_corpora,
        user_sub=state["user_sub"],
        tier=state["tier"],
    )
    return {"chunks": chunks}


@meter_llm
def plan_action(state: AdminOpsState) -> dict:
    """Use the LLM to determine what admin action to take.

    The LLM responds with:
      TOOL: <tool_name> ARGS: <json_args>
    """
    llm = get_llm()
    ctx = "\n".join(c["text"] for c in state.get("chunks", []))
    cfg = get_tier_config(state["tier"])
    tools_str = ", ".join(cfg.allowed_tools)
    prompt = (
        f"Context:\n{ctx}\n\n"
        f"Admin request: {state['message']}\n\n"
        f"Determine the action to take. Respond with exactly one line:\n"
        f"  TOOL: <tool_name> ARGS: <json_args>\n"
        f"Available tools: {tools_str}\n"
        f"If no action is needed, respond with TOOL: none"
    )
    resp = llm.invoke(prompt)
    content = resp.content if hasattr(resp, "content") else str(resp)

    # Parse tool call from response.
    # Format: "TOOL: <tool_name> ARGS: <json_args>"
    planned_tool = None
    tool_args = {}
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("TOOL:"):
            rest = stripped[5:].strip()  # everything after "TOOL:"
            # Split on " ARGS:" to separate tool name from args.
            if " ARGS:" in rest:
                tool_part, args_part = rest.split(" ARGS:", 1)
                planned_tool = tool_part.strip()
                args_str = args_part.strip()
                import json
                try:
                    tool_args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, ValueError):
                    tool_args = {}
            else:
                planned_tool = rest.strip()
            break

    return {"planned_tool": planned_tool, "tool_args": tool_args, "tool_result": {"planned_action": content}}


def maybe_hitl(state: AdminOpsState) -> Command:
    """HITL gate: interrupt before executing ANY write tool.

    Write tools (trigger_scraper, save_numbers) require human approval.
    Read-only tools (query_audit_log, read_token_usage) proceed directly.
    """
    planned_tool = state.get("planned_tool")

    if planned_tool and planned_tool != "none" and is_write_tool(planned_tool):
        decision = interrupt({
            "planned_tool": planned_tool,
            "tool_args": state.get("tool_args", {}),
            "prompt": f"Agent wants to call write tool '{planned_tool}'. "
                      f"This will modify data. Approve?",
        })

        if isinstance(decision, dict) and decision.get("approved"):
            edited = decision.get("edited")
            if edited:
                import json
                try:
                    new_args = json.loads(edited)
                    return Command(update={"tool_args": new_args}, goto="execute")
                except (json.JSONDecodeError, ValueError):
                    pass
            return Command(update={}, goto="execute")
        else:
            return Command(
                update={"response": f"Action '{planned_tool}' rejected by admin reviewer."},
                goto=END,
            )

    # Read tool or no tool — proceed to execute (execute handles "none").
    return Command(update={}, goto="execute")


def execute(state: AdminOpsState) -> dict:
    """Execute the planned admin action."""
    planned_tool = state.get("planned_tool")
    tool_args = state.get("tool_args", {})
    claims = TokenClaims(
        sub=state["user_sub"],
        tier=state["tier"],
        roles=[],
        raw_token=state.get("jwt_token", ""),
    )

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
            user_sub=state["user_sub"],
            jwt_token=state.get("jwt_token", ""),
            category=tool_args.get("category", "default"),
            numbers=tool_args.get("numbers", []),
            will_be=tool_args.get("will_be"),
        )
    else:
        result = {"status": "unknown_action", "message": f"Unknown tool: {planned_tool}"}

    return {"tool_result": result, "response": str(result)}


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
