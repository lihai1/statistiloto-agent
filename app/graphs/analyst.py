"""Analyst worker subgraph.

Multi-step RAG + autonomous analysis with HITL on write tool calls.

Used by paid and admin tiers. The LLM drafts an analysis and may decide
to call tools. If a planned tool writes/updates/deletes data (e.g.
save_numbers), the graph pauses for human approval before executing.
Read-only tools (generate_form, get_statistics, analyze) execute
without HITL.
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
from app.graphs.common import (
    format_history,
    format_ui_context,
    append_history,
    make_retrieve_node,
    make_hitl_gate,
)
from app.graphs.tool_parser import parse_tool_call

log = logging.getLogger(__name__)


class AnalystState(TypedDict):
    user_sub: str
    tier: str
    session_id: str
    message: str
    jwt_token: str
    chunks: list
    history: list
    context: Optional[dict]           # structured UI context
    draft: Optional[str]
    planned_tool: Optional[str]       # tool name the LLM decided to call
    tool_args: Optional[dict]         # arguments for the planned tool
    tool_result: Optional[dict]
    response: Optional[str]


retrieve_docs = make_retrieve_node("analyst")


@meter_llm
def draft_analysis(state: AnalystState) -> dict:
    """Draft an analysis using the global LLM with retrieved context + conversation history.

    The LLM may decide to call a tool. If so, it responds with:
      TOOL: <tool_name> ARGS: {"key": "value", ...}
    Otherwise it responds with a plain text analysis.
    """
    user_sub = state["user_sub"]
    session_id = state["session_id"]
    hist_len = len(state.get("history", []))
    log.info("[analyst.draft] START user=%s session=%s history_len=%d", user_sub, session_id, hist_len)
    try:
        llm = get_llm()
        ctx = "\n".join(c["text"] for c in state.get("chunks", []))
        cfg = get_tier_config(state["tier"])
        tools_str = ", ".join(cfg.allowed_tools)
        hist = format_history(state.get("history", []))
        ui_context = format_ui_context(state.get("context"))
        prompt = (
            f"{SYSTEM_PROMPT_WITH_TOOLS}\n\n"
            f"Available tools for this user: {tools_str}\n\n"
            f"Context:\n{ctx}\n\n"
            f"{ui_context}"
            f"{hist}"
            f"Question: {state['message']}\n\n"
            f"If the user wants you to call a tool, output EXACTLY ONE LINE:\n"
            f"  TOOL: <tool_name> ARGS: <json_args>\n"
            f"If no tool is needed, output a plain text analysis only."
        )
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)

        planned_tool, tool_args = parse_tool_call(content)
        if planned_tool and planned_tool.lower() == "none":
            planned_tool = "none"

        log.info("[analyst.draft] SUCCESS user=%s session=%s planned_tool=%s", user_sub, session_id, planned_tool)
        return {
            "draft": content,
            "planned_tool": planned_tool,
            "tool_args": tool_args,
            "_usage": getattr(resp, "usage_metadata", None),
            "_response_metadata": getattr(resp, "response_metadata", None),
        }
    except Exception as e:
        log.error("[analyst.draft] ERROR user=%s session=%s msg=%s", user_sub, session_id, e, exc_info=True)
        raise


def _analyst_interrupt_fields(state: AnalystState) -> dict:
    return {"draft": state.get("draft", "")}


maybe_hitl = make_hitl_gate("analyst", execute_node="execute_tool", finalize_node="finalize",
                            extra_interrupt_fields=_analyst_interrupt_fields)


def execute_tool(state: AnalystState) -> dict:
    """Execute the planned tool (read or write, already approved if write)."""
    planned_tool = state.get("planned_tool")
    tool_args = state.get("tool_args", {})
    jwt_token = state.get("jwt_token", "")
    user_sub = state["user_sub"]
    session_id = state["session_id"]

    if not planned_tool:
        return {"tool_result": None}

    log.info("[analyst.execute] START user=%s session=%s tool=%s args=%s", user_sub, session_id, planned_tool, tool_args)
    try:
        result = _call_tool(planned_tool, tool_args, user_sub, jwt_token)
        log.info("[analyst.execute] SUCCESS user=%s session=%s tool=%s", user_sub, session_id, planned_tool)
        return {"tool_result": result}
    except Exception as e:
        log.error("[analyst.execute] ERROR user=%s session=%s tool=%s msg=%s", user_sub, session_id, planned_tool, e, exc_info=True)
        raise


def finalize(state: AnalystState) -> dict:
    """Set the final response from the draft, incorporating tool results if any.
    Also appends the current exchange to conversation history for the checkpointer.

    When a tool result is present, the LLM is invoked to transform the structured
    result into concise natural language (per the grounding rules in the system prompt).
    """
    user_sub = state["user_sub"]
    session_id = state["session_id"]
    draft = state.get("draft", "")
    tool_result = state.get("tool_result")

    if tool_result:
        # Invoke the LLM to format the tool result into readable text.
        try:
            llm = get_llm()
            hist = format_history(state.get("history", []))
            import json
            prompt = (
                f"{SYSTEM_PROMPT_WITH_TOOLS}\n\n"
                f"{hist}"
                f"User question: {state['message']}\n\n"
                f"Tool result (JSON):\n{json.dumps(tool_result, ensure_ascii=False, indent=2)}\n\n"
                f"Transform this tool result into a concise, readable natural language response. "
                f"Follow the language, grounding, and formatting rules from the system prompt. "
                f"Do NOT dump raw JSON. Do NOT fabricate data not in the tool result. "
                f"If the tool result contains an error, say you couldn't retrieve the data."
            )
            resp = llm.invoke(prompt)
            response = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            log.error("[analyst.finalize] LLM formatting failed: %s — using raw result", e)
            response = f"{draft}\n\nTool result: {tool_result}"
    else:
        response = draft

    updated_history = append_history(state, response)
    log.info("[analyst.finalize] SUCCESS user=%s session=%s response_len=%d", user_sub, session_id, len(response))
    return {"response": response, "history": updated_history}


def _call_tool(tool_name: str, args: dict, user_sub: str, jwt_token: str) -> dict:
    """Dispatch a tool call by name."""
    from app.tools import lottery_grpc, saved_numbers

    if tool_name == "generate_form":
        return lottery_grpc.generate_form(
            how_many=args.get("how_many", 1),
            form_type=args.get("form_type", 6),
            will_be=args.get("will_be"),
            strength=args.get("strength", 2),
            window_from=args.get("window_from"),
            window_to=args.get("window_to"),
        )
    elif tool_name == "get_statistics":
        return lottery_grpc.get_statistics(
            how_many=args.get("how_many", 10),
            group_size=args.get("group_size", args.get("form_type", 2)),
            strength=args.get("strength", "hot"),
            window_from=args.get("window_from"),
            window_to=args.get("window_to"),
        )
    elif tool_name == "analyze":
        return lottery_grpc.analyze(
            form=args.get("form", []),
            window_from=args.get("window_from"),
            window_to=args.get("window_to"),
        )
    elif tool_name == "list_saved_numbers":
        return saved_numbers.list_saved_numbers(user_sub=user_sub, jwt_token=jwt_token)
    elif tool_name == "save_numbers":
        return saved_numbers.save_numbers(
            user_sub=user_sub,
            jwt_token=jwt_token,
            category=args.get("category", "default"),
            numbers=args.get("numbers", []),
            will_be=args.get("will_be"),
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def build_analyst_graph():
    """Build the analyst worker subgraph."""
    g = StateGraph(AnalystState)
    g.add_node("retrieve", retrieve_docs)
    g.add_node("draft", draft_analysis)
    g.add_node("hitl_gate", maybe_hitl)
    g.add_node("execute_tool", execute_tool)
    g.add_node("finalize", finalize)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "hitl_gate")
    g.add_edge("execute_tool", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
