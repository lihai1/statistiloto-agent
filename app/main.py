"""FastAPI application — agent service entrypoint.

Endpoints:
  POST /chat          — SSE stream of agent events (or 202 paused for HITL)
  POST /approve       — resume a paused HITL thread
  GET  /healthz       — health check
  GET  /llm-config    — read current global LLM config (any authenticated user)
  PUT  /llm-config    — update global LLM config (admin only, hot-reloaded)
  GET  /llm-models    — list available models for a provider (admin only)
  GET  /sessions      — list the caller's chat sessions (tier-limited)
  GET  /sessions/{id} — load a session's full message history
  DELETE /sessions/{id} — delete a session and its checkpointer state
  DELETE /sessions    — delete all of the caller's sessions
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, ConfigDict

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
    context: dict | None = None  # structured UI context (page, numbers, groupSize, etc.)


class ApproveRequest(BaseModel):
    session_id: str
    approved: bool
    edited: str | None = None


class LLMConfigRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider: str        # "ollama" | "gemini" | "openai" | "anthropic" | "mock"
    model: str
    name: str | None = None
    base_url: str | None = Field(default=None, alias="baseUrl")
    api_key: str | None = Field(default=None, alias="apiKey")
    request_timeout_seconds: int | None = Field(default=None, alias="requestTimeoutSeconds")


# ── Graph singleton (built lazily) ───────────────────────────

_graph = None


def get_graph():
    """Get or build the supervisor graph with PostgresSaver checkpointer.

    If the checkpointer's connection was closed and recreated, the graph
    is rebuilt with the new checkpointer instance.
    """
    global _graph
    from app.checkpointer import get_checkpointer

    # get_checkpointer() may recreate the checkpointer if the connection
    # was closed. We need to rebuild the graph in that case.
    checkpointer = get_checkpointer()
    if _graph is not None:
        # Check if the checkpointer was recreated (different instance)
        current_cp = getattr(_graph, "checkpointer", None)
        if current_cp is not checkpointer:
            log.info("Checkpointer changed — rebuilding supervisor graph")
            _graph = None
        else:
            return _graph

    from app.graphs.supervisor import build_supervisor_graph
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
        log.error("[chat] JWT validation failed: %s", e)
        raise HTTPException(status_code=401, detail=str(e))

    log.info("[chat] START user=%s tier=%s session=%s intent=%s", claims.sub, claims.tier, req.session_id, req.intent)

    graph = get_graph()
    thread_id = f"{claims.sub}:{req.session_id}"
    # Pass recursion_limit from tier config — caps graph super-steps to prevent infinite loops.
    from app.config.settings import get_tier_config
    tier_cfg = get_tier_config(claims.tier)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": tier_cfg.recursion_limit,
    }

    # Read conversation history from the previous checkpoint (if any).
    # The checkpointer persists state per thread_id across requests, so
    # the agent can remember prior turns within the same session.
    try:
        prev_state = graph.get_state(config)
        history = list(prev_state.values.get("history", [])) if prev_state and prev_state.values else []
    except Exception:
        history = []

    log.info("[chat] History loaded user=%s session=%s history_len=%d", claims.sub, req.session_id, len(history))

    state = {
        "user_sub": claims.sub,
        "tier": claims.tier,
        "session_id": req.session_id,
        "message": req.message,
        "intent": req.intent,
        "jwt_token": claims.raw_token,
        "history": history,
        "context": req.context,
    }

    import asyncio

    def _run_graph():
        return graph.invoke(state, config)

    try:
        result = await asyncio.to_thread(_run_graph)
    except Exception as e:
        log.error("[chat] Graph invoke ERROR user=%s session=%s msg=%s", claims.sub, req.session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Check if the graph paused for HITL (interrupt).
    # LangGraph stores interrupt info in the state under __interrupt__.
    if isinstance(result, dict) and "__interrupt__" in result:
        log.info("[chat] PAUSED for HITL user=%s session=%s thread=%s", claims.sub, req.session_id, thread_id)
        # Record the session even when paused — the user message counts as a turn.
        _record_session(claims, req, history)
        return {"paused": True, "thread_id": thread_id}

    response = result.get("response") if isinstance(result, dict) else None
    log.info("[chat] SUCCESS user=%s session=%s response_len=%d", claims.sub, req.session_id, len(response) if response else 0)
    # Record/update the session metadata so it appears in history.
    _record_session(claims, req, history)
    return {"response": response, "thread_id": thread_id}


def _record_session(claims, req, prior_history) -> None:
    """Upsert the chat session row. Best-effort — never fails the request."""
    try:
        from app.sessions import upsert_session
        # Title = first user message (when this is the first turn).
        title = req.message[:80] if not prior_history else ""
        upsert_session(
            user_sub=claims.sub,
            session_id=req.session_id,
            title=title,
            last_message=req.message[:200],
            tier=claims.tier,
        )
    except Exception as e:
        log.warning("[chat] Failed to record session metadata: %s", e)


@app.post("/approve")
async def approve(req: ApproveRequest, authorization: str = Header(...)):
    """Resume a paused HITL thread with a human decision."""
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        log.error("[approve] JWT validation failed: %s", e)
        raise HTTPException(status_code=401, detail=str(e))

    log.info("[approve] START user=%s session=%s approved=%s", claims.sub, req.session_id, req.approved)

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
        log.error("[approve] Resume ERROR user=%s session=%s msg=%s", claims.sub, req.session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    response = result.get("response") if isinstance(result, dict) else None
    log.info("[approve] SUCCESS user=%s session=%s approved=%s response_len=%d", claims.sub, req.session_id, req.approved, len(response) if response else 0)
    return {"response": response}


@app.get("/llm-config")
async def get_llm_config(authorization: str = Header(...)):
    """Read the current global LLM config (any authenticated user)."""
    try:
        validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))

    cfg = get_llm_store().get_config()
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "api_key": cfg.api_key,
        "request_timeout_seconds": cfg.request_timeout_seconds,
    }


@app.put("/llm-config")
async def set_llm_config(req: LLMConfigRequest, authorization: str = Header(...)):
    """Create a new LLM config and activate it (admin only). Hot-reloaded within ~10s.

    Each PUT inserts a new named config row and marks it as the active one.
    The previous active config is deactivated (only one active at a time).
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.store import get_pool
    import time
    timeout = req.request_timeout_seconds or 300
    name = req.name or f"{req.provider}/{req.model}"
    pool = get_pool()
    with pool.connection() as conn:
        # Deactivate all existing configs, then insert the new active one.
        conn.execute("UPDATE agent.llm_config SET is_active = FALSE WHERE is_active = TRUE")
        conn.execute(
            """INSERT INTO agent.llm_config
                   (name, provider, model, base_url, api_key, request_timeout_seconds, is_active, updated_by, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, TRUE, %s, %s)
               RETURNING id""",
            (name, req.provider, req.model, req.base_url, req.api_key, timeout, claims.sub, time.time()),
        )

    # Force immediate refresh instead of waiting for the poller.
    get_llm_store().force_refresh()

    return {
        "status": "accepted",
        "provider": req.provider,
        "model": req.model,
        "name": name,
        "request_timeout_seconds": timeout,
        "note": "Hot-reloaded — no restart needed",
    }


@app.get("/llm-configs")
async def list_llm_configs(authorization: str = Header(...)):
    """List all saved LLM configs (admin only). The active one is flagged."""
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.store import get_pool
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT id, name, provider, model, base_url, api_key, request_timeout_seconds, is_active, updated_at
               FROM agent.llm_config
               ORDER BY is_active DESC, updated_at DESC"""
        ).fetchall()

    configs = []
    for r in rows:
        configs.append({
            "id": r[0],
            "name": r[1],
            "provider": r[2],
            "model": r[3],
            "base_url": r[4] or "",
            "api_key": r[5] or "",
            "request_timeout_seconds": r[6],
            "is_active": r[7],
            "updated_at": r[8],
        })
    return {"configs": configs}


@app.post("/llm-configs")
async def create_llm_config(req: LLMConfigRequest, authorization: str = Header(...)):
    """Create a new saved LLM config without activating it (admin only).

    Use PUT /llm-config/{id}/activate to make it the active config.
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.store import get_pool
    import time
    timeout = req.request_timeout_seconds or 300
    name = req.name or f"{req.provider}/{req.model}"
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            """INSERT INTO agent.llm_config
                   (name, provider, model, base_url, api_key, request_timeout_seconds, is_active, updated_by, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, FALSE, %s, %s)
               RETURNING id""",
            (name, req.provider, req.model, req.base_url, req.api_key, timeout, claims.sub, time.time()),
        ).fetchone()
        config_id = row[0] if row else None

    return {"status": "created", "id": config_id, "name": name}


@app.put("/llm-configs/{config_id}/activate")
async def activate_llm_config(config_id: int, authorization: str = Header(...)):
    """Activate a saved LLM config by id (admin only). Hot-reloads the LLM."""
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.store import get_pool
    import time
    pool = get_pool()
    with pool.connection() as conn:
        # Deactivate all, then activate the selected one.
        conn.execute("UPDATE agent.llm_config SET is_active = FALSE WHERE is_active = TRUE")
        row = conn.execute(
            "UPDATE agent.llm_config SET is_active = TRUE, updated_at = %s WHERE id = %s RETURNING id",
            (time.time(), config_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="LLM config not found")

    get_llm_store().force_refresh()
    return {"status": "activated", "id": config_id}


@app.post("/llm-configs/{config_id}/test")
async def test_llm_config(config_id: int, authorization: str = Header(...)):
    """Test that a saved LLM config can actually connect and respond (admin only).

    Builds a temporary LLM instance from the saved config (without activating it)
    and sends a minimal ping message. Returns ok/error with the response or error message.
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.store import get_pool
    from app.llm.config_store import LLMConfig, build_llm
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT provider, model, base_url, api_key, request_timeout_seconds "
            "FROM agent.llm_config WHERE id = %s",
            (config_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="LLM config not found")
        cfg = LLMConfig(
            provider=row[0],
            model=row[1],
            base_url=row[2] or "",
            api_key=row[3] or "",
            request_timeout_seconds=row[4] or 300,
        )

    try:
        llm = build_llm(cfg)
        from langchain_core.messages import HumanMessage
        resp = await llm.ainvoke([HumanMessage(content="ping")])
        text = resp.content if hasattr(resp, "content") else str(resp)
        return {"status": "ok", "id": config_id, "response": text[:200]}
    except Exception as e:
        log.warning("[llm-configs.test] config_id=%s failed: %s", config_id, e)
        return {"status": "error", "id": config_id, "error": str(e)[:500]}


@app.delete("/llm-configs/{config_id}")
async def delete_llm_config(config_id: int, authorization: str = Header(...)):
    """Delete a saved LLM config (admin only). Cannot delete the active config."""
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.store import get_pool
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT is_active FROM agent.llm_config WHERE id = %s",
            (config_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="LLM config not found")
        if row[0]:
            raise HTTPException(status_code=400, detail="Cannot delete the active config")
        conn.execute("DELETE FROM agent.llm_config WHERE id = %s", (config_id,))

    return {"status": "deleted", "id": config_id}


@app.get("/token-usage")
async def get_token_usage(authorization: str = Header(...)):
    """Read token usage stats (admin only). Returns aggregated rows from the DB."""
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.tools.admin_ops import read_token_usage
    rows = read_token_usage(claims)
    return {"rows": rows}


@app.get("/audit-log")
async def get_audit_log(authorization: str = Header(...), limit: int = 50):
    """Read audit log entries (admin only). Returns recent entries from the DB."""
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.tools.admin_ops import query_audit_log
    rows = query_audit_log(claims, limit=limit)
    return {"rows": rows}


@app.post("/reindex")
async def reindex_docs(authorization: str = Header(...)):
    """Reindex product docs into the RAG 'docs' corpus (admin only).

    Reads markdown files from app/rag/docs_source/, embeds them, and inserts
    into agent.embeddings. Content-hash dedup skips unchanged chunks.
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.ingest import ingest_docs
    try:
        result = ingest_docs()
        log.info("[reindex] admin=%s result=%s", claims.sub, result)
        return {"status": "ok", **result}
    except Exception as e:
        log.error("[reindex] failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")


# ── LLM model listing ────────────────────────────────────────

# Known static model lists per provider. Ollama is queried live via its
# /api/tags endpoint (admin-supplied base_url); gemini has a fixed set.
_GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

_OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o3-mini",
]

_ANTHROPIC_MODELS = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]


@app.get("/llm-models")
async def list_llm_models(provider: str, authorization: str = Header(...)):
    """List available models for a provider (admin only).

    For Ollama, queries the live Ollama server's /api/tags endpoint using
    the base_url from the current LLM config (or the `base_url` query param).
    For Gemini, OpenAI, and Anthropic, returns a static list of known models.
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    if provider == "ollama":
        return {"provider": "ollama", "models": await _list_ollama_models(authorization)}
    if provider == "gemini":
        return {"provider": "gemini", "models": list(_GEMINI_MODELS)}
    if provider == "openai":
        return {"provider": "openai", "models": list(_OPENAI_MODELS)}
    if provider == "anthropic":
        return {"provider": "anthropic", "models": list(_ANTHROPIC_MODELS)}
    if provider == "mock":
        return {"provider": "mock", "models": ["fake-list"]}
    raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


async def _list_ollama_models(authorization: str) -> list[str]:
    """Query the Ollama server for installed models via /api/tags."""
    import httpx
    cfg = get_llm_store().get_config()
    base_url = cfg.base_url or "http://ollama:11434"
    base_url = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return [m for m in models if m]
    except Exception as e:
        log.warning("[llm-models] Failed to query Ollama at %s: %s", base_url, e)
        return []


# ── Chat session history ─────────────────────────────────────

@app.get("/sessions")
async def list_sessions(authorization: str = Header(...)):
    """List the caller's chat sessions (newest first).

    Tier limits are enforced on write (upsert), so the returned list is
    already within quota. Also returns the tier's max session count.
    """
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))

    from app.sessions import list_sessions as _list, get_session_limit
    sessions = _list(claims.sub, claims.tier)
    limit = get_session_limit(claims.tier)
    return {
        "sessions": [s.to_dict() for s in sessions],
        "limit": limit,  # None = unlimited
        "tier": claims.tier,
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, authorization: str = Header(...)):
    """Load a session's full message history from the checkpointer."""
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))

    from app.sessions import get_session_messages
    graph = get_graph()
    messages = get_session_messages(claims.sub, session_id, graph)
    return {"session_id": session_id, "messages": messages}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, authorization: str = Header(...)):
    """Delete a single chat session and its checkpointer state."""
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))

    from app.sessions import delete_session as _delete
    deleted = _delete(claims.sub, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.delete("/sessions")
async def delete_all_sessions(authorization: str = Header(...)):
    """Delete all of the caller's chat sessions."""
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))

    from app.sessions import delete_all_sessions as _delete_all
    count = _delete_all(claims.sub)
    return {"status": "deleted", "count": count}
