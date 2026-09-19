"""Shared helpers for LangGraph worker subgraphs.

Centralizes patterns duplicated across nl_assistant, analyst, and admin_ops:
  - conversation history formatting
  - UI context formatting
  - document retrieval node factory
  - history append + truncation
  - HITL approval gate factory
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from langgraph.types import interrupt, Command

from app.config.settings import get_tier_config
from app.rag.retriever import retrieve
from app.tools.registry import is_write_tool

log = logging.getLogger(__name__)

# Maximum messages retained in conversation history (10 user + 10 assistant exchanges).
HISTORY_CAP = 20


def emit_step(node: str) -> None:
    """Emit a step-started progress event to the stream writer.

    Nodes call this as their first statement so /chat/stream can relay a
    ``{"event": "progress", "node": <name>}`` event when the step *starts*
    (not after it completes, which is all stream_mode="updates" gives).
    No-op when the graph runs without a stream writer (plain invoke).
    """
    try:
        from langgraph.config import get_stream_writer
        get_stream_writer()({"event": "progress", "node": node})
    except Exception:
        pass


def format_history(history: list) -> str:
    """Format conversation history for inclusion in the LLM prompt."""
    if not history:
        return ""
    lines = []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"


def format_ui_context(context: dict | None) -> str:
    """Format structured UI context for the LLM prompt.

    The UI sends context like:
      {"page": "statistics", "groupSize": 2, "ordering": "hot", "archiveWindow": {"lastDraws": 100}}
      {"page": "analyze", "numbers": [7,11,17,24,31,36]}
    """
    if not context:
        return ""
    return f"User's current page context:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"


def append_history(state: dict, response: str) -> list:
    """Append the current user message + assistant response to history, capped at HISTORY_CAP."""
    updated = list(state.get("history", []))
    updated.append({"role": "user", "content": state["message"]})
    updated.append({"role": "assistant", "content": response})
    if len(updated) > HISTORY_CAP:
        updated = updated[-HISTORY_CAP:]
    return updated


def make_retrieve_node(log_prefix: str) -> Callable[[dict], dict]:
    """Factory: create a retrieve_docs node with a given log prefix (e.g. 'analyst', 'admin_ops').

    The returned function retrieves from the tier's allowed corpora using the
    user's message as the query.

    Phase 4 — Intent-aware retrieval:
      - trivial / out_of_scope / statistics / admin_operation (high confidence) → skip RAG (0 chunks)
      - domain_explanation → docs corpus only (top_k=3)
      - ambiguous / medium confidence → docs corpus only (top_k=3)
    This avoids unnecessary embedding queries and DB round-trips for paths
    where RAG adds no value.
    """
    def retrieve_docs(state: dict) -> dict:
        emit_step("retrieve")
        user_sub = state["user_sub"]
        tier = state["tier"]
        session_id = state["session_id"]
        lang = state.get("lang")
        message = state.get("message", "")
        log.info("[%s.retrieve] START user=%s tier=%s session=%s lang=%s", log_prefix, user_sub, tier, session_id, lang)
        try:
            # Phase 4: Determine if RAG is needed based on message content.
            from app.normalizer import normalize as _normalize
            req = _normalize(
                message=message,
                lang_hint=lang,
                context=state.get("context"),
                conversation=None,
                client_intent_hint=state.get("client_intent_hint"),
            )

            # Skip RAG for deterministic paths where it adds no value.
            if req.request_kind in ("trivial", "out_of_scope"):
                log.info("[%s.retrieve] SKIP kind=%s — deterministic path, 0 RAG", log_prefix, req.request_kind)
                return {"chunks": []}

            # Statistics and admin operations with high confidence → structured
            # data from tools, no RAG needed.
            # confidence is a float [0,1]; high = >= 0.8
            if req.request_kind in ("statistics", "number_analysis", "form_generation") and req.confidence >= 0.8:
                log.info("[%s.retrieve] SKIP kind=%s confidence=%.2f — tool data, 0 RAG", log_prefix, req.request_kind, req.confidence)
                return {"chunks": []}

            if req.request_kind == "admin_operation" and req.confidence >= 0.8:
                log.info("[%s.retrieve] SKIP kind=admin_operation confidence=%.2f — typed tool, 0 RAG", log_prefix, req.confidence)
                return {"chunks": []}

            # domain_explanation and ambiguous/medium → docs only, top_k=3.
            cfg = get_tier_config(tier)
            # Only retrieve from docs (and examples for domain explanations).
            corpora = ["docs"]
            if "examples" in cfg.rag_corpora and req.request_kind == "domain_explanation":
                corpora.append("examples")

            chunks = retrieve(
                query=message,
                corpora=corpora,
                user_sub=user_sub,
                tier=tier,
                lang=lang,
                top_k=3,
            )
            log.info("[%s.retrieve] SUCCESS user=%s session=%s kind=%s chunks=%d", log_prefix, user_sub, session_id, req.request_kind, len(chunks))
            return {"chunks": chunks}
        except Exception as e:
            log.error("[%s.retrieve] ERROR user=%s session=%s msg=%s", log_prefix, user_sub, session_id, e, exc_info=True)
            raise
    return retrieve_docs


def make_hitl_gate(
    log_prefix: str,
    execute_node: str,
    finalize_node: Optional[str] = None,
    extra_interrupt_fields: Optional[Callable[[dict], dict]] = None,
) -> Callable[[dict], Command]:
    """Factory: create a HITL gate node.

    Args:
        log_prefix: e.g. 'analyst', 'admin_ops' — used in log messages.
        execute_node: name of the node to goto when a tool is approved or is a read tool.
        finalize_node: name of the node to goto when no tool is planned (analyst only).
                       If None (admin_ops), a no-tool plan still goes to execute_node.
        extra_interrupt_fields: optional callable returning extra fields for the interrupt payload
                                (e.g. analyst includes the draft).

    The gate interrupts for any write tool. Read tools proceed directly. If no tool
    is planned, it routes to finalize_node (if provided) or execute_node.
    """
    def hitl_gate(state: dict) -> Command:
        planned_tool = state.get("planned_tool")
        user_sub = state["user_sub"]
        session_id = state["session_id"]

        if planned_tool and planned_tool != "none" and is_write_tool(planned_tool):
            log.info("[%s.hitl] PAUSE user=%s session=%s tool=%s — awaiting approval",
                     log_prefix, user_sub, session_id, planned_tool)
            interrupt_payload = {
                "planned_tool": planned_tool,
                "tool_args": state.get("tool_args", {}),
                "prompt": f"Agent wants to call write tool '{planned_tool}'. This will modify data. Approve?",
            }
            if extra_interrupt_fields:
                interrupt_payload.update(extra_interrupt_fields(state))
            decision = interrupt(interrupt_payload)

            if isinstance(decision, dict) and decision.get("approved"):
                log.info("[%s.hitl] APPROVED user=%s session=%s tool=%s", log_prefix, user_sub, session_id, planned_tool)
                edited = decision.get("edited")
                if edited:
                    try:
                        new_args = json.loads(edited)
                        return Command(update={"tool_args": new_args}, goto=execute_node)
                    except (json.JSONDecodeError, ValueError):
                        pass
                return Command(update={}, goto=execute_node)
            else:
                log.info("[%s.hitl] REJECTED user=%s session=%s tool=%s", log_prefix, user_sub, session_id, planned_tool)
                return Command(
                    update={"response": f"Tool call '{planned_tool}' rejected by reviewer."},
                    goto=END if finalize_node is None else END,
                )

        # Read tool or no tool.
        if planned_tool and planned_tool != "none":
            log.info("[%s.hitl] READ tool proceeds user=%s session=%s tool=%s",
                     log_prefix, user_sub, session_id, planned_tool)
            return Command(update={}, goto=execute_node)

        # No tool at all.
        log.info("[%s.hitl] No tool, proceeding user=%s session=%s", log_prefix, user_sub, session_id)
        if finalize_node:
            return Command(update={}, goto=finalize_node)
        return Command(update={}, goto=execute_node)

    return hitl_gate


# Re-export END for convenience in factory closures.
from langgraph.graph import END  # noqa: E402
