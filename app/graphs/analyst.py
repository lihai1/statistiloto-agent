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
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

from app.config.settings import get_tier_config
from app.llm.router import get_llm
from app.metering import meter_llm
from app.rag.retriever import retrieve
from app.tools.registry import is_write_tool

log = logging.getLogger(__name__)


class AnalystState(TypedDict):
    user_sub: str
    tier: str
    session_id: str
    message: str
    jwt_token: str
    chunks: list
    draft: Optional[str]
    planned_tool: Optional[str]       # tool name the LLM decided to call
    tool_args: Optional[dict]         # arguments for the planned tool
    tool_result: Optional[dict]
    response: Optional[str]


def retrieve_docs(state: AnalystState) -> dict:
    """Retrieve relevant docs from the tier's allowed corpora."""
    cfg = get_tier_config(state["tier"])
    chunks = retrieve(
        query=state["message"],
        corpora=cfg.rag_corpora,
        user_sub=state["user_sub"],
        tier=state["tier"],
    )
    return {"chunks": chunks}


@meter_llm
def draft_analysis(state: AnalystState) -> dict:
    """Draft an analysis using the global LLM with retrieved context.

    The LLM may decide to call a tool. If so, it responds with:
      TOOL: <tool_name> ARGS: {"key": "value", ...}
    Otherwise it responds with a plain text analysis.
    """
    llm = get_llm()
    ctx = "\n".join(c["text"] for c in state.get("chunks", []))
    cfg = get_tier_config(state["tier"])
    tools_str = ", ".join(cfg.allowed_tools)
    prompt = (
        f"Context:\n{ctx}\n\n"
        f"Question: {state['message']}\n\n"
        f"Provide a detailed analysis. If you need to call a tool, "
        f"respond with exactly one line:\n"
        f"  TOOL: <tool_name> ARGS: <json_args>\n"
        f"Available tools: {tools_str}\n"
        f"Otherwise, provide your analysis as plain text."
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
                if planned_tool.lower() == "none":
                    planned_tool = "none"
            break

    return {"draft": content, "planned_tool": planned_tool, "tool_args": tool_args}


def maybe_hitl(state: AnalystState) -> Command:
    """HITL gate: interrupt before executing ANY write tool.

    If the LLM planned a write tool (save_numbers, etc.), pause for
    human approval. Read-only tools and plain-text responses proceed.
    """
    planned_tool = state.get("planned_tool")

    if planned_tool and is_write_tool(planned_tool):
        decision = interrupt({
            "draft": state.get("draft", ""),
            "planned_tool": planned_tool,
            "tool_args": state.get("tool_args", {}),
            "prompt": f"Agent wants to call write tool '{planned_tool}'. Approve?",
        })

        if isinstance(decision, dict) and decision.get("approved"):
            edited = decision.get("edited")
            if edited:
                import json
                try:
                    new_args = json.loads(edited)
                    return Command(update={"tool_args": new_args}, goto="execute_tool")
                except (json.JSONDecodeError, ValueError):
                    pass
            return Command(update={}, goto="execute_tool")
        else:
            return Command(
                update={"response": f"Tool call '{planned_tool}' rejected by reviewer."},
                goto=END,
            )

    # No write tool planned — if there's a read tool, execute it; otherwise finalize.
    if planned_tool and not is_write_tool(planned_tool):
        return Command(update={}, goto="execute_tool")

    # No tool at all — go straight to finalize.
    return Command(update={}, goto="finalize")


def execute_tool(state: AnalystState) -> dict:
    """Execute the planned tool (read or write, already approved if write)."""
    planned_tool = state.get("planned_tool")
    tool_args = state.get("tool_args", {})
    jwt_token = state.get("jwt_token", "")
    user_sub = state["user_sub"]

    if not planned_tool:
        return {"tool_result": None}

    result = _call_tool(planned_tool, tool_args, user_sub, jwt_token)
    return {"tool_result": result}


def finalize(state: AnalystState) -> dict:
    """Set the final response from the draft, incorporating tool results if any."""
    draft = state.get("draft", "")
    tool_result = state.get("tool_result")

    if tool_result:
        return {"response": f"{draft}\n\nTool result: {tool_result}"}
    return {"response": draft}


def _call_tool(tool_name: str, args: dict, user_sub: str, jwt_token: str) -> dict:
    """Dispatch a tool call by name."""
    from app.tools import lottery_grpc, saved_numbers

    if tool_name == "generate_form":
        return lottery_grpc.generate_form(
            how_many=args.get("how_many", 1),
            form_type=args.get("form_type", 0),
            will_be=args.get("will_be"),
            strength=args.get("strength", 2),
        )
    elif tool_name == "get_statistics":
        return lottery_grpc.get_statistics(
            how_many=args.get("how_many", 10),
            form_type=args.get("form_type", 0),
            strength=args.get("strength", 2),
        )
    elif tool_name == "analyze":
        return lottery_grpc.analyze(form=args.get("form", []))
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
