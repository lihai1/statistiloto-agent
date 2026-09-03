"""FastAPI application — agent service entrypoint.

Endpoints:
  POST /chat          — SSE stream of agent events (or 202 paused for HITL)
  POST /approve       — resume a paused HITL thread
  GET  /healthz       — health check
  GET  /llm-config    — read current global LLM config (any authenticated user)
  PUT  /llm-config    — update global LLM config (admin only, hot-reloaded)
  GET  /llm-models    — list available models for a provider (admin only)
  GET  /free-llm      — read the free-tier LLM toggle (admin only)
  PUT  /free-llm      — set the free-tier LLM toggle (admin only)
  GET  /sessions      — list the caller's chat sessions (tier-limited)
  GET  /sessions/{id} — load a session's full message history
  DELETE /sessions/{id} — delete a session and its checkpointer state
  DELETE /sessions    — delete all of the caller's sessions
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, ConfigDict

from app.config.settings import get_settings
from app.llm.config_store import get_llm_store, set_llm_store, LLMConfigStore, LLMConfig, build_llm
from app.llm.router import set_llm_override, reset_llm_override
from app.security import validate_jwt, require_admin, TokenClaims, JWTError

log = logging.getLogger(__name__)

app = FastAPI(title="statistiloto-agent", version="0.1.0")

# ── Request/Response models ──────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    intent: str | None = None
    context: dict | None = None  # structured UI context (page, numbers, groupSize, etc.)
    config_id: int | None = None  # admin-only: override the active LLM for this request
    lang: str | None = None  # user language code ("en" | "he"); filters examples corpus


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


class FreeLlmToggleRequest(BaseModel):
    enabled: bool


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

    # Auto-ingest RAG corpora (docs + examples) on startup.
    # Runs in a background thread so the HTTP server starts immediately.
    # Fault-tolerant: failures are logged but never block startup —
    # admin can retry via POST /reindex once Ollama is ready.
    if s.rag.auto_ingest:
        import asyncio
        asyncio.create_task(_run_startup_ingestion())


async def _run_startup_ingestion():
    """Ingest docs + examples in a background thread on startup.

    Uses asyncio.to_thread because the ingestion code is synchronous
    (psycopg pool + Ollama embeddings client). Failures are logged but
    never crash the agent — the admin /reindex endpoint can retry.
    """
    import asyncio
    from app.rag.ingest import ingest_docs, ingest_examples
    try:
        docs_result = await asyncio.to_thread(ingest_docs)
        log.info("[startup] docs ingestion: %s", docs_result)
    except Exception as e:
        log.warning("[startup] docs ingestion failed (retry via /reindex): %s", e)
    try:
        examples_result = await asyncio.to_thread(ingest_examples)
        log.info("[startup] examples ingestion: %s", examples_result)
    except Exception as e:
        log.warning("[startup] examples ingestion failed (retry via /reindex): %s", e)


@app.on_event("shutdown")
async def shutdown():
    store = get_llm_store()
    store.stop_poller()
    log.info("Agent service stopped")


# ── Endpoints ────────────────────────────────────────────────

@app.get("/healthz")
async def health():
    return {"status": "ok"}


def _read_llm_config_by_id(config_id: int) -> LLMConfig | None:
    """Read a single llm_config row by id from the DB. Returns None if not found."""
    from app.rag.store import get_pool
    try:
        pool = get_pool()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT provider, model, base_url, api_key, request_timeout_seconds "
                "FROM agent.llm_config WHERE id = %s",
                (config_id,),
            ).fetchone()
            if row:
                return LLMConfig(
                    provider=row[0],
                    model=row[1],
                    base_url=row[2] or "",
                    api_key=row[3] or "",
                    request_timeout_seconds=row[4] or 300,
                )
    except Exception as e:
        log.warning("[chat] Failed to read config_id=%s: %s", config_id, e)
    return None


@app.post("/chat")
async def chat(req: ChatRequest, authorization: str = Header(...)):
    """Process a chat message. Returns JSON with response or paused status.

    Uses sync graph.invoke() because PostgresSaver doesn't implement
    async checkpoint methods in this version of langgraph-checkpoint-postgres.
    The sync invoke runs in a thread pool via asyncio.to_thread.

    Admin users can pass ``config_id`` to override the active LLM for this
    single request. The override is scoped to the request via a ContextVar
    that propagates into the graph's thread.
    """
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        log.error("[chat] JWT validation failed: %s", e)
        raise HTTPException(status_code=401, detail=str(e))

    log.info("[chat] START user=%s tier=%s session=%s intent=%s", claims.sub, claims.tier, req.session_id, req.intent)

    # Admin-only: build a per-request LLM override from the specified config_id.
    override_token = None
    if req.config_id is not None:
        if claims.tier != "admin":
            log.warning("[chat] Non-admin user=%s attempted config_id override — ignored", claims.sub)
        else:
            cfg = _read_llm_config_by_id(req.config_id)
            if cfg is None:
                raise HTTPException(status_code=404, detail=f"LLM config {req.config_id} not found")
            try:
                override_llm = build_llm(cfg)
                override_token = set_llm_override(override_llm)
                log.info("[chat] LLM override config_id=%s provider=%s model=%s", req.config_id, cfg.provider, cfg.model)
            except Exception as e:
                log.warning("[chat] Failed to build override LLM config_id=%s: %s — using global", req.config_id, e)

    try:
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
            # Phase 6: load conversation state for follow-up detection.
            prev_conv_state = prev_state.values.get("conversation_state") if prev_state and prev_state.values else None
        except Exception:
            history = []
            prev_conv_state = None

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
            "lang": (req.lang or "").strip().lower() or None,
            # Client intent is an untrusted hint — logged for audit, never authoritative.
            "client_intent_hint": req.intent,
            # Phase 6: pass previous conversation state for follow-up inheritance.
            "prev_conversation_state": prev_conv_state,
        }

        import asyncio

        def _run_graph():
            return graph.invoke(state, config)

        try:
            result = await asyncio.to_thread(_run_graph)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            log.error("[chat] Graph invoke ERROR user=%s session=%s type=%s msg=%s",
                      claims.sub, req.session_id, error_type, error_msg, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Agent error: {error_type}: {error_msg}",
            )

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
    finally:
        # Always clear the override, even on error.
        if override_token is not None:
            reset_llm_override(override_token)


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


# ── SSE streaming chat ────────────────────────────────────────

# Human-readable labels for graph node names (for progress events).
_NODE_LABELS = {
    "supervisor": "Routing request",
    "retrieve_context": "Retrieving context",
    "draft_analysis": "Planning analysis",
    "execute_tool": "Executing tool",
    "finalize": "Finalizing response",
    "plan_action": "Planning admin action",
    "nl_general": "Generating response",
}


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_stream_state(claims, req, history, prev_conv_state) -> dict:
    """Build the graph input state shared by /chat and /chat/stream."""
    return {
        "user_sub": claims.sub,
        "tier": claims.tier,
        "session_id": req.session_id,
        "message": req.message,
        "intent": req.intent,
        "jwt_token": claims.raw_token,
        "history": history,
        "context": req.context,
        "lang": (req.lang or "").strip().lower() or None,
        "client_intent_hint": req.intent,
        "prev_conversation_state": prev_conv_state,
    }


def _setup_chat_request(claims, req):
    """Common setup for /chat and /chat/stream: graph, thread_id, config, history."""
    graph = get_graph()
    thread_id = f"{claims.sub}:{req.session_id}"
    from app.config.settings import get_tier_config
    tier_cfg = get_tier_config(claims.tier)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": tier_cfg.recursion_limit,
    }
    try:
        prev_state = graph.get_state(config)
        history = list(prev_state.values.get("history", [])) if prev_state and prev_state.values else []
        prev_conv_state = prev_state.values.get("conversation_state") if prev_state and prev_state.values else None
    except Exception:
        history = []
        prev_conv_state = None
    return graph, thread_id, config, history, prev_conv_state


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, authorization: str = Header(...)):
    """Stream agent events as SSE (node-level progress + final response).

    Same JWT validation, LLM override, history, and recursion-limit behavior
    as POST /chat. Emits:
      - event: progress  {node, label}   — after each graph node completes
      - event: done      {response, thread_id}  — final response
      - event: paused    {thread_id}     — HITL interrupt (write tool)
      - event: error     {message}       — on failure

    The existing POST /chat remains unchanged as a compatibility/fallback endpoint.
    """
    try:
        claims = validate_jwt(authorization)
    except JWTError as e:
        log.error("[chat.stream] JWT validation failed: %s", e)
        raise HTTPException(status_code=401, detail=str(e))

    log.info("[chat.stream] START user=%s tier=%s session=%s", claims.sub, claims.tier, req.session_id)

    # Admin-only LLM override (same as /chat).
    override_token = None
    if req.config_id is not None:
        if claims.tier != "admin":
            log.warning("[chat.stream] Non-admin user=%s attempted config_id override — ignored", claims.sub)
        else:
            cfg = _read_llm_config_by_id(req.config_id)
            if cfg is None:
                raise HTTPException(status_code=404, detail=f"LLM config {req.config_id} not found")
            try:
                override_llm = build_llm(cfg)
                override_token = set_llm_override(override_llm)
            except Exception as e:
                log.warning("[chat.stream] Failed to build override LLM: %s — using global", e)

    async def _stream_generator():
        """Yield SSE events from graph.stream()."""
        import asyncio
        try:
            graph, thread_id, config, history, prev_conv_state = _setup_chat_request(claims, req)
            state = _build_stream_state(claims, req, history, prev_conv_state)

            def _run_stream():
                """Run graph.stream() in a thread (sync checkpointer)."""
                chunks = []
                for chunk in graph.stream(state, config, stream_mode="updates"):
                    chunks.append(chunk)
                return chunks

            chunks = await asyncio.to_thread(_run_stream)

            # Emit progress events for each node update.
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                for node_name in chunk:
                    label = _NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
                    yield _sse_event("progress", {"node": node_name, "label": label})

            # Check final state for HITL interrupt or response.
            final_state = graph.get_state(config)
            if final_state and final_state.next:
                # Graph paused — HITL interrupt.
                log.info("[chat.stream] PAUSED user=%s session=%s thread=%s", claims.sub, req.session_id, thread_id)
                _record_session(claims, req, history)
                yield _sse_event("paused", {"thread_id": thread_id})
            else:
                values = final_state.values if final_state else {}
                response = values.get("response") if isinstance(values, dict) else None
                log.info("[chat.stream] DONE user=%s session=%s response_len=%d",
                         claims.sub, req.session_id, len(response) if response else 0)
                _record_session(claims, req, history)
                yield _sse_event("done", {"response": response, "thread_id": thread_id})

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            log.error("[chat.stream] ERROR user=%s session=%s type=%s msg=%s",
                      claims.sub, req.session_id, error_type, error_msg, exc_info=True)
            yield _sse_event("error", {"message": f"{error_type}: {error_msg}"})
        finally:
            if override_token is not None:
                reset_llm_override(override_token)

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
        error_type = type(e).__name__
        error_msg = str(e)
        log.error("[approve] Resume ERROR user=%s session=%s type=%s msg=%s",
                  claims.sub, req.session_id, error_type, error_msg, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {error_type}: {error_msg}",
        )

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


@app.put("/llm-configs/{config_id}")
async def update_llm_config(config_id: int, req: LLMConfigRequest, authorization: str = Header(...)):
    """Update an existing saved LLM config by id (admin only).

    Does not change activation status. If the updated config is the active
    one, the LLM is hot-reloaded with the new values.
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
            """UPDATE agent.llm_config
                   SET name = %s, provider = %s, model = %s, base_url = %s,
                       api_key = %s, request_timeout_seconds = %s,
                       updated_by = %s, updated_at = %s
                 WHERE id = %s RETURNING id, is_active""",
            (name, req.provider, req.model, req.base_url, req.api_key, timeout, claims.sub, time.time(), config_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Config {config_id} not found")
        was_active = row[1]

    # Hot-reload if the updated config is the active one.
    if was_active:
        try:
            get_llm_store().force_refresh()
            log.info("[llm-configs.update] config_id=%s was active — LLM hot-reloaded", config_id)
        except Exception as e:
            log.warning("[llm-configs.update] hot-reload failed: %s", e)

    return {"status": "updated", "id": config_id, "name": name}


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
    """Reindex product docs AND examples into RAG corpora (admin only).

    Reads markdown from app/rag/docs_source/ into the 'docs' corpus and
    YAML examples from app/rag/examples_source/<lang>/ into the 'examples'
    corpus. Content-hash dedup skips unchanged chunks.
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.rag.ingest import ingest_docs, ingest_examples
    try:
        docs_result = ingest_docs()
        examples_result = ingest_examples()
        log.info("[reindex] admin=%s docs=%s examples=%s",
                 claims.sub, docs_result, examples_result)
        return {"status": "ok", "docs": docs_result, "examples": examples_result}
    except Exception as e:
        log.error("[reindex] failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")


# ── Free-tier LLM toggle ──────────────────────────────────────

@app.get("/free-llm")
async def get_free_llm_toggle(authorization: str = Header(...)):
    """Read the free-tier LLM toggle state (admin only).

    Returns whether free-tier users are currently allowed to invoke the LLM
    for ambiguous/domain-explanation requests. Default: disabled.
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.free_tier_llm import is_free_llm_enabled
    return {"enabled": is_free_llm_enabled()}


@app.put("/free-llm")
async def set_free_llm_toggle(req: FreeLlmToggleRequest, authorization: str = Header(...)):
    """Set the free-tier LLM toggle (admin only).

    When enabled, free-tier ambiguous/domain-explanation requests route to
    the nl_assistant worker and invoke the LLM. When disabled (default), they
    receive a generic deterministic response with zero LLM calls.

    This toggle is process-scoped (in-memory) and does NOT persist across
    restarts. Set the FREE_LLM_ENABLED env var for persistent defaults.

    Security: this toggle only controls LLM invocation for ambiguous/domain
    requests — it does NOT grant free users any additional tools or write
    access. Authorization remains enforced by CapabilityConfig.
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    from app.free_tier_llm import set_free_llm_enabled, is_free_llm_enabled
    set_free_llm_enabled(req.enabled)
    log.info("[free-llm] admin=%s set enabled=%s", claims.sub, req.enabled)
    return {"enabled": is_free_llm_enabled(), "updated_by": claims.sub}


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
async def list_llm_models(
    provider: str,
    authorization: str = Header(...),
    base_url: str | None = Query(None),
):
    """List available models for a provider (admin only).

    For Ollama, queries the live Ollama server's /api/tags endpoint.
    The base URL is resolved in priority order:
      1. the ``base_url`` query param (sent by the UI from the form field),
      2. the active LLM config's ``base_url``,
      3. the default ``http://ollama:11434``.
    This ensures Ollama models are fetched from the correct server even when
    a non-Ollama config is currently active.
    For Gemini, OpenAI, and Anthropic, returns a static list of known models
    with an empty ``capabilities`` array (capabilities are not surfaced for
    non-Ollama providers).

    Response shape::

        {"provider": "ollama",
         "models": [{"name": "llama3.1:8b", "size": 4661214619, "capabilities": ["tools"]}, ...]}
    """
    try:
        claims = validate_jwt(authorization)
        require_admin(claims)
    except JWTError as e:
        raise HTTPException(status_code=403, detail=str(e))

    if provider == "ollama":
        return {"provider": "ollama", "models": await _list_ollama_models(base_url)}
    if provider == "gemini":
        return {"provider": "gemini", "models": [{"name": m, "size": 0, "capabilities": []} for m in _GEMINI_MODELS]}
    if provider == "openai":
        return {"provider": "openai", "models": [{"name": m, "size": 0, "capabilities": []} for m in _OPENAI_MODELS]}
    if provider == "anthropic":
        return {"provider": "anthropic", "models": [{"name": m, "size": 0, "capabilities": []} for m in _ANTHROPIC_MODELS]}
    if provider == "mock":
        return {"provider": "mock", "models": [{"name": "fake-list", "size": 0, "capabilities": []}]}
    raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")


def _parse_ollama_tags(data: dict) -> list[dict]:
    """Parse Ollama ``/api/tags`` JSON into ``[{name, size, capabilities}]``.

    Pure helper (no I/O) so it can be unit-tested without a live Ollama.
    The ``capabilities`` array is taken verbatim from Ollama's response
    (Ollama >= 0.30.0 reports it on ``/api/tags``); older servers that omit
    it get an empty list per model. ``size`` is the model size in bytes
    (0 when absent) — surfaced so the UI can show it greyed-out next to
    the model name.
    """
    models = []
    for m in data.get("models", []):
        name = m.get("name", "") or m.get("model", "")
        if not name:
            continue
        caps = m.get("capabilities")
        if not isinstance(caps, list):
            caps = []
        try:
            size = int(m.get("size", 0) or 0)
        except (TypeError, ValueError):
            size = 0
        models.append({"name": name, "size": size, "capabilities": [str(c) for c in caps]})
    return models


async def _list_ollama_models(base_url: str | None = None) -> list[dict]:
    """Query the Ollama server for installed models via /api/tags.

    ``base_url`` resolution priority:
      1. the explicit ``base_url`` argument (from the UI form),
      2. the active LLM config's ``base_url``,
      3. the default ``http://ollama:11434``.
    """
    import httpx
    if not base_url:
        cfg = get_llm_store().get_config()
        base_url = cfg.base_url or "http://ollama:11434"
    base_url = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/api/tags")
            resp.raise_for_status()
            return _parse_ollama_tags(resp.json())
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
