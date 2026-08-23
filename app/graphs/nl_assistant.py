"""NL Assistant worker subgraph.

Single/multi-turn natural-language → tool calls.
Used by free tier (docs-only RAG) and as a fallback for downgraded intents.
"""

from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from app.config.settings import get_tier_config
from app.llm.router import get_llm
from app.metering import meter_llm
from app.rag.retriever import retrieve

log = logging.getLogger(__name__)


class NLAssistantState(TypedDict):
    user_sub: str
    tier: str
    session_id: str
    message: str
    chunks: list
    history: list
    response: Optional[str]


def retrieve_docs(state: NLAssistantState) -> dict:
    """Retrieve relevant docs (role-scoped corpora)."""
    user_sub = state["user_sub"]
    tier = state["tier"]
    session_id = state["session_id"]
    log.info("[nl_assistant.retrieve] START user=%s tier=%s session=%s", user_sub, tier, session_id)
    try:
        cfg = get_tier_config(tier)
        chunks = retrieve(
            query=state["message"],
            corpora=cfg.rag_corpora,
            user_sub=user_sub,
            tier=tier,
        )
        log.info("[nl_assistant.retrieve] SUCCESS user=%s session=%s chunks=%d", user_sub, session_id, len(chunks))
        return {"chunks": chunks}
    except Exception as e:
        log.error("[nl_assistant.retrieve] ERROR user=%s session=%s msg=%s", user_sub, session_id, e, exc_info=True)
        raise


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
        hist = _format_history(state.get("history", []))
        prompt = f"Context:\n{ctx}\n\n{hist}Question: {state['message']}\n\nAnswer:"
        resp = llm.invoke(prompt)
        content = resp.content if hasattr(resp, "content") else str(resp)
        # Append the current exchange to history so the checkpointer persists it.
        updated_history = list(state.get("history", []))
        updated_history.append({"role": "user", "content": state["message"]})
        updated_history.append({"role": "assistant", "content": content})
        # Cap at 20 messages (10 exchanges) to avoid unbounded growth.
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]
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


def _format_history(history: list) -> str:
    """Format conversation history for inclusion in the LLM prompt."""
    if not history:
        return ""
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"
