"""Supervisor graph — top-level hierarchical router.

Routes by tier + intent to one of three worker subgraphs:
  - nl_assistant: NL → tool calls (free/paid/admin)
  - analyst: multi-step RAG + analysis (paid/admin)
  - admin_ops: admin operations (admin only)

Tier gating happens HERE — the single choke point. Disallowed intents
are downgraded to nl_assistant.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from app.graphs.nl_assistant import build_nl_assistant_graph
from app.graphs.analyst import build_analyst_graph
from app.graphs.admin_ops import build_admin_ops_graph

log = logging.getLogger(__name__)


class SupervisorState(TypedDict):
    user_sub: str
    tier: str                       # free | paid | admin
    session_id: str
    message: str
    intent: Optional[Literal["nl_assistant", "analyst", "admin_ops"]]
    jwt_token: str
    history: list
    context: Optional[dict]         # structured UI context (page, numbers, groupSize, etc.)
    # Worker output bubbles up here.
    response: Optional[str]
    chunks: list
    draft: Optional[str]
    planned_tool: Optional[str]     # tool name the LLM decided to call
    tool_args: Optional[dict]       # arguments for the planned tool
    tool_result: Optional[dict]


def classify_intent(state: SupervisorState) -> str:
    """Route by tier + intent. Tier gating happens HERE — the single choke point."""
    tier = state.get("tier", "free")
    intent = state.get("intent")
    user_sub = state.get("user_sub", "unknown")
    session_id = state.get("session_id", "")

    # Admin-only intents are unreachable for non-admin tiers.
    if intent == "admin_ops" and tier != "admin":
        log.info("[supervisor.classify] DOWNGRADE admin_ops→nl_assistant user=%s tier=%s session=%s", user_sub, tier, session_id)
        return "nl_assistant"

    # Paid-only intents are unreachable for free tier.
    if intent == "analyst" and tier == "free":
        log.info("[supervisor.classify] DOWNGRADE analyst→nl_assistant user=%s tier=%s session=%s", user_sub, tier, session_id)
        return "nl_assistant"

    route = intent or "nl_assistant"
    log.info("[supervisor.classify] ROUTE user=%s tier=%s session=%s intent=%s → %s", user_sub, tier, session_id, intent, route)
    return route


def build_supervisor_graph(checkpointer=None):
    """Build the top-level supervisor graph with three worker subgraphs.

    Args:
        checkpointer: a LangGraph checkpointer (PostgresSaver for production,
                      None for unit tests).
    """
    # Workers are compiled subgraphs — each owns its state schema, tools, RAG filter.
    nl_graph = build_nl_assistant_graph()
    analyst_graph = build_analyst_graph()
    admin_graph = build_admin_ops_graph()

    g = StateGraph(SupervisorState)
    g.add_node("nl_assistant", nl_graph)
    g.add_node("analyst", analyst_graph)
    g.add_node("admin_ops", admin_graph)

    g.add_conditional_edges(START, classify_intent, {
        "nl_assistant": "nl_assistant",
        "analyst": "analyst",
        "admin_ops": "admin_ops",
    })
    g.add_edge("nl_assistant", END)
    g.add_edge("analyst", END)
    g.add_edge("admin_ops", END)

    return g.compile(checkpointer=checkpointer)
