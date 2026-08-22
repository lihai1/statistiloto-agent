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
    response: Optional[str]


def retrieve_docs(state: NLAssistantState) -> dict:
    """Retrieve relevant docs (role-scoped corpora)."""
    cfg = get_tier_config(state["tier"])
    chunks = retrieve(
        query=state["message"],
        corpora=cfg.rag_corpora,
        user_sub=state["user_sub"],
        tier=state["tier"],
    )
    return {"chunks": chunks}


@meter_llm
def generate_response(state: NLAssistantState) -> dict:
    """Generate a response using the global LLM with retrieved context."""
    llm = get_llm()
    ctx = "\n".join(c["text"] for c in state.get("chunks", []))
    prompt = f"Context:\n{ctx}\n\nQuestion: {state['message']}\n\nAnswer:"
    resp = llm.invoke(prompt)
    content = resp.content if hasattr(resp, "content") else str(resp)
    return {"response": content}


def build_nl_assistant_graph():
    """Build the NL assistant worker subgraph (no checkpointer — supervisor has it)."""
    g = StateGraph(NLAssistantState)
    g.add_node("retrieve", retrieve_docs)
    g.add_node("generate", generate_response)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()
