"""NL Assistant worker subgraph.

Single/multi-turn natural-language → tool calls.
Used by free tier (docs-only RAG) and as a fallback for downgraded intents.
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from app.llm.router import get_llm
from app.metering import meter_llm
from app.prompt_builder import build_prompt
from app.graphs.common import format_history, append_history, make_retrieve_node

log = logging.getLogger(__name__)


class NLAssistantState(TypedDict):
    user_sub: str
    tier: str
    session_id: str
    message: str
    chunks: list
    history: list
    context: Optional[dict]
    lang: Optional[str]
    response: Optional[str]


retrieve_docs = make_retrieve_node("nl_assistant")


@meter_llm
def generate_response(state: NLAssistantState) -> dict:
    """Generate a response using the global LLM with retrieved context + conversation history."""
    user_sub = state["user_sub"]
    session_id = state["session_id"]
    hist_len = len(state.get("history", []))
    log.info("[nl_assistant.generate] START user=%s session=%s history_len=%d", user_sub, session_id, hist_len)
    try:
        llm = get_llm()
        ctx = "\n".join(c["text"] for c in state.get("chunks", []))
        hist = format_history(state.get("history", []))
        lang = state.get("lang") or "en"

        # Phase 5: use compact route-specific prompt.
        prompt = build_prompt(
            route="nl_general",
            language=lang,
            user_message=state["message"],
            knowledge=ctx,
            history=hist,
        )
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        updated_history = append_history(state, content)
        log.info("[nl_assistant.generate] SUCCESS user=%s session=%s response_len=%d", user_sub, session_id, len(content))
        return {
            "response": content,
            "history": updated_history,
            "_usage": getattr(resp, "usage_metadata", None),
            "_response_metadata": getattr(resp, "response_metadata", None),
        }
    except Exception as e:
        log.error("[nl_assistant.generate] ERROR user=%s session=%s msg=%s", user_sub, session_id, e, exc_info=True)
        raise


def build_nl_assistant_graph():
    """Build the NL assistant worker subgraph (no checkpointer — supervisor has it)."""
    g = StateGraph(NLAssistantState)
    g.add_node("retrieve", retrieve_docs)
    g.add_node("generate", generate_response)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
