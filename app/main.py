"""FastAPI application — agent service entrypoint.

Endpoints:
  POST /chat          — SSE stream of agent events (or 202 paused for HITL)
  POST /approve       — resume a paused HITL thread
  GET  /healthz       — health check
  GET  /llm-config    — read current global LLM config (any authenticated user)
  PUT  /llm-config    — update global LLM config (admin only, hot-reloaded)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

from app.config.settings import get_settings
from app.llm.config_store import get_llm_store, set_llm_store, LLMConfigStore
from app.security import validate_jwt, require_admin, TokenClaims, JWTError

log = logging.getLogger(__name__)

app = FastAPI(title="statistiloto-agent", version="0.1.0")

# ── Request/Response models ──────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    intent: str | None = None


class ApproveRequest(BaseModel):
    session_id: str
    approved: bool
    edited: str | None = None


class LLMConfigRequest(BaseModel):
    provider: str        # "ollama" | "gemini" | "mock"
    model: str
    base_url: str | None = None
    api_key: str | None = None


# ── Graph singleton (built lazily) ───────────────────────────

_graph = None


def get_graph():
    """Get or build the supervisor graph with PostgresSaver checkpointer."""
    global _graph
    if _graph is not None:
        return _graph

    from app.graphs.supervisor import build_supervisor_graph
    from app.checkpointer import get_checkpointer

    checkpointer = get_checkpointer()
    _graph = build_supervisor_graph(checkpointer=checkpointer)
    log.info("Supervisor graph compiled with PostgresSaver checkpointer")
    return _graph


def set_graph(graph):
    """Replace the graph (for testing)."""
    global _graph
    _graph = graph


def reset_graph():
    """Reset the graph so next get_graph() builds a fresh one."""
    global _graph
    _graph = None


# ── Startup ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    s = get_settings()
    logging.basicConfig(level=getattr(logging, s.logging.level, logging.INFO))
    # Start the LLM config poller for hot-reload.
    store = get_llm_store()
    store.start_poller()
    log.info("Agent service started (LLM provider=%s model=%s)",
             store.get_config().provider, store.get_config().model)


@app.on_event("shutdown")
async def shutdown():
    store = get_llm_store()
    store.stop_poller()
    log.info("Agent service stopped")


# ── Endpoints ────────────────────────────────────────────────

@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest, authorization: str = Header(...)):
    """Process a chat message. Returns JSON with response or paused status.

    Uses sync graph.invoke() because PostgresSaver doesn't implement
    async checkpoint methods in this version of langgraph-checkpoint-postgres.
    The sync invoke runs in a thread pool via asyncio.to_thread.
    """
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))

    graph = get_graph()
    thread_id = f"{claims.sub}:{req.session_id}"
    # Pass recursion_limit from tier config — caps graph super-steps to prevent infinite loops.
    from app.config.settings import get_tier_config
    tier_cfg = get_tier_config(claims.tier)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": tier_cfg.recursion_limit,
    }
    state = {
        "user_sub": claims.sub,
        "tier": claims.tier,
        "session_id": req.session_id,
        "message": req.message,
        "intent": req.intent,
        "jwt_token": claims.raw_token,
        "history": [],
    }

    import asyncio

    def _run_graph():
        return graph.invoke(state, config)

    try:
        result = await asyncio.to_thread(_run_graph)
    except Exception as e:
        log.error("Graph invoke error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Check if the graph paused for HITL (interrupt).
    # LangGraph stores interrupt info in the state under __interrupt__.
    if isinstance(result, dict) and "__interrupt__" in result:
        return {"paused": True, "thread_id": thread_id}

    response = result.get("response") if isinstance(result, dict) else None
    return {"response": response, "thread_id": thread_id}


@app.post("/approve")
async def approve(req: ApproveRequest, authorization: str = Header(...)):
    """Resume a paused HITL thread with a human decision."""
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))

    graph = get_graph()
    thread_id = f"{claims.sub}:{req.session_id}"
    from app.config.settings import get_tier_config
    tier_cfg = get_tier_config(claims.tier)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": tier_cfg.recursion_limit,
    }

    resume_value = {"approved": req.approved, "edited": req.edited}

    import asyncio

    def _resume_graph():
        return graph.invoke(Command(resume=resume_value), config)

    try:
        result = await asyncio.to_thread(_resume_graph)
    except Exception as e:
        log.error("Approve error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    response = result.get("response") if isinstance(result, dict) else None
    return {"response": response}


@app.get("/llm-config")
async def get_llm_config(authorization: str = Header(...)):
    """Read the current global LLM config (any authenticated user)."""
    try:
        validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))

    cfg = get_llm_store().get_config()
    return {"provider": cfg.provider, "model": cfg.model}


@app.put("/llm-config")
async def set_llm_config(req: LLMConfigRequest, authorization: str = Header(...)):
    """Update the global LLM config (admin only). Hot-reloaded within ~10s."""
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.store import get_pool
    import time
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO agent.llm_config (provider, model, base_url, api_key, updated_by, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (req.provider, req.model, req.base_url, req.api_key, claims.sub, time.time()),
        )

    # Force immediate refresh instead of waiting for the poller.
    get_llm_store().force_refresh()

    return {
        "status": "accepted",
        "provider": req.provider,
        "model": req.model,
        "note": "Hot-reloaded — no restart needed",
    }
