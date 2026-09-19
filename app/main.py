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
  POST   /sessions/archive  — archive all of the caller's sessions (soft-delete)
  GET    /sessions/archived — list archived sessions (admin only)
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, ConfigDict

from app.config.settings import get_settings
from app.llm.config_store import get_llm_store, set_llm_store, LLMConfigStore, LLMConfig, build_llm, MASKED_API_KEY, mask_api_key
from app.llm.router import set_llm_override, reset_llm_override
from app.security import (
    TokenClaims,
    get_current_user,
    require_admin_user,
)

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
    num_predict: int | None = Field(default=None, alias="numPredict")  # max tokens to generate
    context_window_size: int | None = Field(default=None, alias="contextWindowSize")  # context window in tokens


class FreeLlmToggleRequest(BaseModel):
    enabled: bool


# ── Graph singleton (built lazily) ───────────────────────────

_graph = None
_graph_lock = asyncio.Lock()


async def get_graph():
    """Get or build the supervisor graph with AsyncPostgresSaver checkpointer.

    If the checkpointer instance changed (e.g. test injection), the graph
    is rebuilt with the new checkpointer instance.
    """
    global _graph
    from app.checkpointer import get_checkpointer

    checkpointer = await get_checkpointer()
    if _graph is not None and getattr(_graph, "checkpointer", None) is checkpointer:
        return _graph

    async with _graph_lock:
        checkpointer = await get_checkpointer()
        if _graph is not None and getattr(_graph, "checkpointer", None) is checkpointer:
            return _graph
        from app.graphs.supervisor import build_supervisor_graph
        _graph = build_supervisor_graph(checkpointer=checkpointer)
        log.info("Supervisor graph compiled with AsyncPostgresSaver checkpointer")
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

    # Ensure trusted local Ollama models exist (pull missing ones in the
    # background — pulls can be GBs and must not block startup).
    if s.llm.provider == "ollama" and s.llm.ollama.models:
        import asyncio
        from app.llm.model_pull import ensure_ollama_models
        asyncio.create_task(ensure_ollama_models(
            s.llm.ollama.base_url, s.llm.ollama.models))

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
    # Drain in-flight stream tasks — publish an error so subscribers
    # see a terminal event instead of a silently dropped connection.
    tasks = list(_stream_tasks)
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    from app.checkpointer import close_checkpointer
    await close_checkpointer()
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
                "SELECT provider, model, base_url, api_key, request_timeout_seconds, num_predict, context_window_size "
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
                    num_predict=row[5],
                    context_window_size=row[6] if len(row) > 6 else None,
                )
    except Exception as e:
        log.warning("[chat] Failed to read config_id=%s: %s", config_id, e)
    return None


def _enforce_budget(claims: TokenClaims) -> str | None:
    """Return a deterministic over-budget message, or None when within budget.

    Checked before ANY graph execution so an over-budget user costs zero
    LLM calls. Fails open (returns None) when the DB check itself errors.
    """
    from app.metering import check_daily_budget
    if not check_daily_budget(claims.sub, claims.tier):
        log.warning("[budget] DENIED user=%s tier=%s — daily budget exceeded", claims.sub, claims.tier)
        return "You've reached your daily usage limit. Please try again tomorrow or contact the administrator."
    return None


@app.post("/chat")
async def chat(req: ChatRequest, claims: TokenClaims = Depends(get_current_user)):
    """Process a chat message. Returns JSON with response or paused status.

    Uses async graph.ainvoke() with AsyncPostgresSaver.

    Admin users can pass ``config_id`` to override the active LLM for this
    single request. The override is scoped to the request via a ContextVar.
    """
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
        graph = await get_graph()
        thread_id = f"{claims.sub}:{req.session_id}"

        # Daily budget check — before any graph work.
        budget_msg = _enforce_budget(claims)
        if budget_msg is not None:
            _record_session(claims, req, [])
            return {"response": budget_msg, "thread_id": thread_id, "budget_exceeded": True}

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
            prev_state = await graph.aget_state(config)
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

        try:
            result = await graph.ainvoke(state, config)
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
            paused_state = None
            try:
                paused_state = await graph.aget_state(config)
            except Exception:
                pass
            return {"paused": True, "thread_id": thread_id,
                    "action": _extract_paused_action(paused_state)}

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
        message = getattr(req, "message", "") or ""
        # Title = first user message (when this is the first turn).
        title = message[:80] if not prior_history else ""
        upsert_session(
            user_sub=claims.sub,
            session_id=req.session_id,
            title=title,
            last_message=message[:200],
            tier=claims.tier,
        )
    except Exception as e:
        log.warning("[chat] Failed to record session metadata: %s", e)


# ── SSE streaming chat ────────────────────────────────────────

# Human-readable labels for graph node names (for progress events).
# Keys match the actual add_node() names across all (sub)graphs.
_NODE_LABELS = {
    "direct_trivial": "Answering",
    "direct_out_of_scope": "Answering",
    "direct_generic": "Answering",
    "direct_clarify": "Asking for clarification",
    "direct_unauthorized": "Checking permissions",
    "direct_multi_request": "Splitting requests",
    "direct_tool": "Executing tool",
    "retrieve": "Retrieving context",
    "draft": "Planning analysis",
    "plan": "Planning action",
    "generate": "Generating response",
    "execute_tool": "Executing tool",
    "execute": "Executing action",
    "finalize": "Finalizing response",
    "nl_assistant": "Assistant",
    "analyst": "Analyst",
    "admin_ops": "Admin operations",
    "supervisor": "Routing request",
    "fetch": "Fetching draw data",
    "insert": "Inserting new draws",
    "prizes": "Backfilling prizes",
}

# Subgraph nodes whose LLM output is the user-facing answer. Token events
# are forwarded only for these — planner/draft text must never leak.
_TOKEN_NODES = {"generate", "finalize"}

# Event names that end a run. Exactly one is emitted per run.
_TERMINAL_EVENTS = frozenset({"done", "paused", "error"})

_HEARTBEAT_SECONDS = 10.0

# Registry of in-flight background stream tasks so shutdown can cancel
# them cleanly (and subscribers get an error event instead of silence).
_stream_tasks: set[asyncio.Task] = set()


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _message_text(chunk) -> str:
    """Extract text content from an AIMessageChunk (str or content blocks)."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _extract_paused_action(final_state) -> dict | None:
    """Pull the HITL interrupt payload (planned_tool/tool_args/prompt).

    LangGraph stores interrupt values on the paused tasks of the state
    snapshot. Returns {tool, args, prompt} or None.
    """
    try:
        for task in getattr(final_state, "tasks", None) or []:
            for intr in getattr(task, "interrupts", None) or []:
                value = getattr(intr, "value", intr)
                if isinstance(value, dict):
                    return {
                        "tool": value.get("planned_tool"),
                        "args": value.get("tool_args", {}),
                        "prompt": value.get("prompt"),
                    }
    except Exception as e:
        log.warning("[stream] Failed to extract paused action: %s", e)
    return None


async def _run_events(graph, input_, config, thread_id):
    """Yield normalized stream event dicts from graph.astream().

    Event dicts (the ``event`` key is the SSE event name / Redis payload type):
      progress {node, label}   — a graph step started (custom stream writer)
      token    {delta}         — LLM answer chunk (generate/finalize only)
      paused   {thread_id, action} — HITL interrupt (terminal)
      done     {response, thread_id} — final answer (terminal)
      error    {message}       — failure (terminal)

    Exactly one terminal event is always yielded last.
    """
    try:
        async for _ns, mode, payload in graph.astream(
            input_,
            config,
            stream_mode=["custom", "messages"],
            subgraphs=True,
        ):
            if mode == "custom":
                if isinstance(payload, dict) and payload.get("event") == "progress":
                    node = payload.get("node") or ""
                    yield {
                        "event": "progress",
                        "node": node,
                        "label": _NODE_LABELS.get(node, node.replace("_", " ").title()),
                    }
            elif mode == "messages":
                chunk, meta = payload if isinstance(payload, tuple) else (payload, {})
                node = (meta or {}).get("langgraph_node") or ""
                if node in _TOKEN_NODES:
                    text = _message_text(chunk)
                    if text:
                        yield {"event": "token", "delta": text}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log.error("[stream] Graph run error thread=%s: %s", thread_id, error_msg, exc_info=True)
        yield {"event": "error", "message": error_msg}
        return

    # Stream finished — inspect the final state for the terminal event.
    try:
        final_state = await graph.aget_state(config)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log.error("[stream] Failed to read final state thread=%s: %s", thread_id, error_msg)
        yield {"event": "error", "message": error_msg}
        return

    if final_state and final_state.next:
        log.info("[stream] PAUSED thread=%s", thread_id)
        yield {
            "event": "paused",
            "thread_id": thread_id,
            "action": _extract_paused_action(final_state),
        }
    else:
        values = final_state.values if final_state else {}
        response = values.get("response") if isinstance(values, dict) else None
        log.info("[stream] DONE thread=%s response_len=%d", thread_id, len(response) if response else 0)
        yield {"event": "done", "response": response, "thread_id": thread_id}


async def _run_events_with_heartbeat(graph, input_, config, thread_id):
    """Wrap _run_events with periodic heartbeat events.

    A heartbeat is yielded every ``_HEARTBEAT_SECONDS`` while waiting for
    the next real event, so relays can distinguish "slow LLM" from "dead
    agent" and clients never sit on a silent stream.
    """
    ait = _run_events(graph, input_, config, thread_id).__aiter__()
    pending: asyncio.Task = asyncio.ensure_future(ait.__anext__())
    try:
        while True:
            done, _pending = await asyncio.wait({pending}, timeout=_HEARTBEAT_SECONDS)
            if not done:
                yield {"event": "heartbeat"}
                continue
            try:
                ev = pending.result()
            except StopAsyncIteration:
                return
            yield ev
            if ev.get("event") in _TERMINAL_EVENTS:
                return
            pending = asyncio.ensure_future(ait.__anext__())
    finally:
        if not pending.done():
            pending.cancel()
        await ait.aclose()


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


async def _setup_chat_request(claims, req):
    """Common setup for /chat and /chat/stream: graph, thread_id, config, history."""
    graph = await get_graph()
    thread_id = f"{claims.sub}:{req.session_id}"
    from app.config.settings import get_tier_config
    tier_cfg = get_tier_config(claims.tier)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": tier_cfg.recursion_limit,
    }
    try:
        prev_state = await graph.aget_state(config)
        history = list(prev_state.values.get("history", [])) if prev_state and prev_state.values else []
        prev_conv_state = prev_state.values.get("conversation_state") if prev_state and prev_state.values else None
    except Exception:
        history = []
        prev_conv_state = None
    return graph, thread_id, config, history, prev_conv_state


def _resolve_override_cfg(req, claims: TokenClaims, endpoint: str):
    """Validate an optional admin config_id override. Returns LLMConfig or None."""
    if getattr(req, "config_id", None) is None:
        return None
    if claims.tier != "admin":
        log.warning("[%s] Non-admin user=%s attempted config_id override — ignored", endpoint, claims.sub)
        return None
    cfg = _read_llm_config_by_id(req.config_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"LLM config {req.config_id} not found")
    return cfg


async def _publish_run(graph, input_, config, thread_id, channel, claims, req, history, override_cfg):
    """Background task: run the graph and publish events to a Redis Stream."""
    from app.redis_client import publish_event
    override_token = None
    if override_cfg is not None:
        try:
            override_token = set_llm_override(build_llm(override_cfg))
        except Exception as e:
            log.warning("[stream] Failed to build override LLM: %s — using global", e)
    try:
        async for ev in _run_events_with_heartbeat(graph, input_, config, thread_id):
            await publish_event(channel, ev)
            if ev.get("event") in _TERMINAL_EVENTS:
                _record_session(claims, req, history)
    except asyncio.CancelledError:
        # Shutdown/cancellation — give subscribers a terminal event.
        await publish_event(channel, {"event": "error", "message": "stream cancelled"})
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log.error("[stream] Publisher error thread=%s: %s", thread_id, error_msg, exc_info=True)
        await publish_event(channel, {"event": "error", "message": error_msg})
    finally:
        if override_token is not None:
            reset_llm_override(override_token)


async def _sse_run(graph, input_, config, thread_id, claims, req, history, override_cfg):
    """Async generator: yield SSE-formatted events for a graph run."""
    override_token = None
    if override_cfg is not None:
        try:
            override_token = set_llm_override(build_llm(override_cfg))
        except Exception as e:
            log.warning("[stream] Failed to build override LLM: %s — using global", e)
    try:
        async for ev in _run_events_with_heartbeat(graph, input_, config, thread_id):
            yield _sse_event(ev["event"], {k: v for k, v in ev.items() if k != "event"})
            if ev.get("event") in _TERMINAL_EVENTS:
                _record_session(claims, req, history)
    except (asyncio.CancelledError, GeneratorExit):
        raise
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        log.error("[stream] SSE error thread=%s: %s", thread_id, error_msg, exc_info=True)
        yield _sse_event("error", {"message": error_msg})
    finally:
        if override_token is not None:
            reset_llm_override(override_token)


def _stream_endpoint_common(req, claims, request: Request, endpoint: str):
    """Shared validation for stream endpoints.

    Returns (override_cfg, use_redis, wants_sse). The Accept header picks
    the transport explicitly:
      - ``Accept: text/event-stream`` → inline SSE
      - ``Accept: application/json``  → Redis Streams relay (503 if down)
      - anything else (e.g. */*)      → Redis if available else inline SSE
    """
    accept = (request.headers.get("accept") or "").lower()
    wants_sse = "text/event-stream" in accept
    wants_json = "application/json" in accept
    return _resolve_override_cfg(req, claims, endpoint), wants_sse, wants_json


async def _start_stream_run(req, claims, request: Request, endpoint: str, input_factory):
    """Shared body for /chat/stream and /approve/stream.

    ``input_factory(graph, config, history, prev_conv_state)`` returns the
    graph input (state dict or Command). Returns either JSON
    {thread_id, channel} or a StreamingResponse.
    """
    from app.redis_client import is_redis_available, stream_key

    override_cfg, wants_sse, wants_json = _stream_endpoint_common(req, claims, request, endpoint)

    # Mode contract: explicit JSON wants the Redis relay — fail fast (503)
    # BEFORE any graph work if Redis is down; explicit SSE → inline.
    # No Accept header defaults to inline SSE — a generic client asking for
    # a stream gets a stream, never an unexpected JSON channel response.
    redis_ok = await is_redis_available()
    if wants_json and not wants_sse and not redis_ok:
        raise HTTPException(status_code=503, detail="Stream relay unavailable (Redis down)")
    use_redis = redis_ok and wants_json and not wants_sse

    graph, thread_id, config, history, prev_conv_state = await _setup_chat_request(claims, req)
    input_ = input_factory(graph, config, history, prev_conv_state)

    if use_redis:
        channel = stream_key(thread_id)
        task = asyncio.create_task(
            _publish_run(graph, input_, config, thread_id, channel, claims, req, history, override_cfg)
        )
        _stream_tasks.add(task)
        task.add_done_callback(_stream_tasks.discard)
        return {"thread_id": thread_id, "channel": channel}

    return StreamingResponse(
        _sse_run(graph, input_, config, thread_id, claims, req, history, override_cfg),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, claims: TokenClaims = Depends(get_current_user)):
    """Stream agent events (progress/token/heartbeat + one terminal event).

    Transport is selected by the Accept header:
      - ``Accept: application/json`` → returns ``{"thread_id", "channel"}``
        and the run publishes events to the Redis Stream
        ``agent:stream:{thread_id}:{run_id}`` (XADD, replayable via XREAD from 0-0).
        Returns 503 before running the graph when Redis is unavailable.
      - ``Accept: text/event-stream`` → inline SSE response.
      - other/absent → Redis relay when available, else inline SSE.

    Exactly one graph execution per request — the response is never re-POSTed.
    """
    log.info("[chat.stream] START user=%s tier=%s session=%s", claims.sub, claims.tier, req.session_id)

    budget_msg = _enforce_budget(claims)
    if budget_msg is not None:
        thread_id = f"{claims.sub}:{req.session_id}"
        _record_session(claims, req, [])
        accept = (request.headers.get("accept") or "").lower()
        if "application/json" in accept and "text/event-stream" not in accept:
            # Publish the deterministic reply to the stream so the BFF sees it.
            from app.redis_client import is_redis_available, publish_event, stream_key
            if not await is_redis_available():
                raise HTTPException(status_code=503, detail="Stream relay unavailable (Redis down)")
            channel = stream_key(thread_id)
            await publish_event(channel, {"event": "done", "response": budget_msg, "thread_id": thread_id})
            return {"thread_id": thread_id, "channel": channel}
        return StreamingResponse(
            _static_sse(budget_msg, thread_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return await _start_stream_run(
        req, claims, request, "chat.stream",
        lambda graph, config, history, prev: _build_stream_state(claims, req, history, prev),
    )


async def _static_sse(message: str, thread_id: str):
    """SSE stream that emits a single deterministic done event (budget etc.)."""
    yield _sse_event("done", {"response": message, "thread_id": thread_id})


@app.post("/approve")
async def approve(req: ApproveRequest, claims: TokenClaims = Depends(get_current_user)):
    """Resume a paused HITL thread with a human decision."""
    log.info("[approve] START user=%s session=%s approved=%s", claims.sub, req.session_id, req.approved)

    budget_msg = _enforce_budget(claims)
    if budget_msg is not None:
        return {"response": budget_msg, "budget_exceeded": True}

    graph = await get_graph()
    thread_id = f"{claims.sub}:{req.session_id}"
    from app.config.settings import get_tier_config
    tier_cfg = get_tier_config(claims.tier)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": tier_cfg.recursion_limit,
    }

    resume_value = {"approved": req.approved, "edited": req.edited}

    try:
        result = await graph.ainvoke(Command(resume=resume_value), config)
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


@app.post("/approve/stream")
async def approve_stream(req: ApproveRequest, request: Request, claims: TokenClaims = Depends(get_current_user)):
    """Resume a paused HITL thread, streaming progress like /chat/stream.

    Same Accept-header transport contract and event schema as /chat/stream.
    """
    log.info("[approve.stream] START user=%s session=%s approved=%s", claims.sub, req.session_id, req.approved)

    budget_msg = _enforce_budget(claims)
    if budget_msg is not None:
        thread_id = f"{claims.sub}:{req.session_id}"
        accept = (request.headers.get("accept") or "").lower()
        if "application/json" in accept and "text/event-stream" not in accept:
            from app.redis_client import is_redis_available, publish_event, stream_key
            if not await is_redis_available():
                raise HTTPException(status_code=503, detail="Stream relay unavailable (Redis down)")
            channel = stream_key(thread_id)
            await publish_event(channel, {"event": "done", "response": budget_msg, "thread_id": thread_id})
            return {"thread_id": thread_id, "channel": channel}
        return StreamingResponse(
            _static_sse(budget_msg, thread_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    resume_value = {"approved": req.approved, "edited": req.edited}
    return await _start_stream_run(
        req, claims, request, "approve.stream",
        lambda graph, config, history, prev: Command(resume=resume_value),
    )


@app.get("/llm-config")
async def get_llm_config(_claims: TokenClaims = Depends(get_current_user)):
    """Read the current global LLM config (any authenticated user)."""
    cfg = get_llm_store().get_config()
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        # Never expose the stored key — masked display value only.
        "api_key": mask_api_key(cfg.api_key),
        "api_key_set": bool(cfg.api_key),
        "request_timeout_seconds": cfg.request_timeout_seconds,
        "num_predict": cfg.num_predict,
        "context_window_size": cfg.context_window_size,
    }


@app.put("/llm-config")
async def set_llm_config(req: LLMConfigRequest, claims: TokenClaims = Depends(require_admin_user)):
    """Create a new LLM config and activate it (admin only). Hot-reloaded within ~10s.

    Each PUT inserts a new named config row and marks it as the active one.
    The previous active config is deactivated (only one active at a time).
    """
    from app.rag.store import get_pool
    import time
    timeout = req.request_timeout_seconds or 300
    num_predict = req.num_predict if req.num_predict is not None and req.num_predict > 0 else 256
    name = req.name or f"{req.provider}/{req.model}"
    pool = get_pool()
    with pool.connection() as conn:
        # A masked api_key means "keep the current key" — read it before
        # deactivating the existing row.
        api_key = req.api_key
        if api_key == MASKED_API_KEY:
            row = conn.execute(
                "SELECT api_key FROM agent.llm_config WHERE is_active = TRUE LIMIT 1"
            ).fetchone()
            api_key = row[0] if row else ""
        # Deactivate all existing configs, then insert the new active one.
        conn.execute("UPDATE agent.llm_config SET is_active = FALSE WHERE is_active = TRUE")
        conn.execute(
            """INSERT INTO agent.llm_config
                   (name, provider, model, base_url, api_key, request_timeout_seconds, num_predict, context_window_size, is_active, updated_by, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s)
               RETURNING id""",
            (name, req.provider, req.model, req.base_url, api_key, timeout, num_predict, req.context_window_size, claims.sub, time.time()),
        )

    # Force immediate refresh instead of waiting for the poller.
    get_llm_store().force_refresh()

    return {
        "status": "accepted",
        "provider": req.provider,
        "model": req.model,
        "name": name,
        "request_timeout_seconds": timeout,
        "num_predict": num_predict,
        "context_window_size": req.context_window_size,
        "note": "Hot-reloaded — no restart needed",
    }


@app.get("/llm-configs")
async def list_llm_configs(claims: TokenClaims = Depends(require_admin_user)):
    """List all saved LLM configs (admin only). The active one is flagged."""
    from app.rag.store import get_pool
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT id, name, provider, model, base_url, api_key, request_timeout_seconds, num_predict, context_window_size, is_active, updated_at
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
            "api_key": mask_api_key(r[5]),
            "api_key_set": bool(r[5]),
            "request_timeout_seconds": r[6],
            "num_predict": r[7],
            "context_window_size": r[8] if len(r) > 8 else None,
            "is_active": r[9] if len(r) > 9 else r[8],
            "updated_at": r[10] if len(r) > 10 else r[9],
        })
    return {"configs": configs}


@app.post("/llm-configs")
async def create_llm_config(req: LLMConfigRequest, claims: TokenClaims = Depends(require_admin_user)):
    """Create a new saved LLM config without activating it (admin only).

    Use PUT /llm-config/{id}/activate to make it the active config.
    """
    from app.rag.store import get_pool
    import time
    timeout = req.request_timeout_seconds or 300
    num_predict = req.num_predict if req.num_predict is not None and req.num_predict > 0 else 256
    name = req.name or f"{req.provider}/{req.model}"
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            """INSERT INTO agent.llm_config
                   (name, provider, model, base_url, api_key, request_timeout_seconds, num_predict, context_window_size, is_active, updated_by, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE, %s, %s)
               RETURNING id""",
            # A masked key on a brand-new config means "no key".
            (name, req.provider, req.model, req.base_url,
             "" if req.api_key == MASKED_API_KEY else req.api_key,
             timeout, num_predict, req.context_window_size, claims.sub, time.time()),
        ).fetchone()
        config_id = row[0] if row else None

    return {"status": "created", "id": config_id, "name": name}


@app.put("/llm-configs/{config_id}")
async def update_llm_config(config_id: int, req: LLMConfigRequest, claims: TokenClaims = Depends(require_admin_user)):
    """Update an existing saved LLM config by id (admin only).

    Does not change activation status. If the updated config is the active
    one, the LLM is hot-reloaded with the new values.
    """

    from app.rag.store import get_pool
    import time
    timeout = req.request_timeout_seconds or 300
    num_predict = req.num_predict if req.num_predict is not None and req.num_predict > 0 else 256
    name = req.name or f"{req.provider}/{req.model}"
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            # NULLIF(masked, masked) → NULL → COALESCE keeps the stored key,
            # so a UI that echoes the mask back doesn't clobber the secret.
            """UPDATE agent.llm_config
                   SET name = %s, provider = %s, model = %s, base_url = %s,
                       api_key = COALESCE(NULLIF(%s, %s), api_key),
                       request_timeout_seconds = %s, num_predict = %s,
                       context_window_size = %s,
                       updated_by = %s, updated_at = %s
                 WHERE id = %s RETURNING id, is_active""",
            (name, req.provider, req.model, req.base_url, req.api_key, MASKED_API_KEY, timeout, num_predict, req.context_window_size, claims.sub, time.time(), config_id),
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
async def activate_llm_config(config_id: int, claims: TokenClaims = Depends(require_admin_user)):
    """Activate a saved LLM config by id (admin only). Hot-reloads the LLM."""
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
async def test_llm_config(config_id: int, claims: TokenClaims = Depends(require_admin_user)):
    """Test that a saved LLM config can actually connect (admin only).

    Uses ``check_connection`` to verify connectivity and credentials
    without sending an inference request (no token cost). Returns
    ``{"status": "ok", "id": ..., "response": detail}`` on success or
    ``{"status": "error", "id": ..., "response": error_msg}`` on failure.
    """
    from app.rag.store import get_pool
    from app.llm.config_store import LLMConfig, check_connection
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT provider, model, base_url, api_key, request_timeout_seconds, num_predict, context_window_size "
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
            num_predict=row[5],
            context_window_size=row[6] if len(row) > 6 else None,
        )

    ok, detail = await check_connection(cfg)
    if ok:
        return {"status": "ok", "id": config_id, "response": detail}
    log.warning("[llm-configs.test] config_id=%s failed: %s", config_id, detail)
    return {"status": "error", "id": config_id, "response": detail}


@app.delete("/llm-configs/{config_id}")
async def delete_llm_config(config_id: int, claims: TokenClaims = Depends(require_admin_user)):
    """Delete a saved LLM config (admin only). Cannot delete the active config."""
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
async def get_token_usage(claims: TokenClaims = Depends(require_admin_user)):
    """Read token usage stats (admin only). Returns aggregated rows from the DB."""
    from app.tools.admin_ops import read_token_usage
    rows = read_token_usage(claims)
    return {"rows": rows}


@app.get("/audit-log")
async def get_audit_log(claims: TokenClaims = Depends(require_admin_user), limit: int = 50):
    """Read audit log entries (admin only). Returns recent entries from the DB."""
    from app.tools.admin_ops import query_audit_log
    rows = query_audit_log(claims, limit=limit)
    return {"rows": rows}


@app.post("/reindex")
async def reindex_docs(claims: TokenClaims = Depends(require_admin_user)):
    """Reindex product docs AND examples into RAG corpora (admin only).

    Reads markdown from app/rag/docs_source/ into the 'docs' corpus and
    YAML examples from app/rag/examples_source/<lang>/ into the 'examples'
    corpus. Content-hash dedup skips unchanged chunks.
    """
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
async def get_free_llm_toggle(claims: TokenClaims = Depends(require_admin_user)):
    """Read the free-tier LLM toggle state (admin only).

    Returns whether free-tier users are currently allowed to invoke the LLM
    for ambiguous/domain-explanation requests. Default: disabled.
    """
    from app.free_tier_llm import is_free_llm_enabled
    return {"enabled": is_free_llm_enabled()}


@app.put("/free-llm")
async def set_free_llm_toggle(req: FreeLlmToggleRequest, claims: TokenClaims = Depends(require_admin_user)):
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
    claims: TokenClaims = Depends(require_admin_user),
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
async def list_sessions(claims: TokenClaims = Depends(get_current_user)):
    """List the caller's chat sessions (newest first).

    Tier limits are enforced on write (upsert), so the returned list is
    already within quota. Also returns the tier's max session count.
    """
    from app.sessions import list_sessions as _list, get_session_limit
    sessions = _list(claims.sub, claims.tier)
    limit = get_session_limit(claims.tier)
    return {
        "sessions": [s.to_dict() for s in sessions],
        "limit": limit,  # None = unlimited
        "tier": claims.tier,
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, claims: TokenClaims = Depends(get_current_user)):
    """Load a session's full message history from the checkpointer."""
    from app.sessions import get_session_messages
    graph = await get_graph()
    messages = await get_session_messages(claims.sub, session_id, graph)
    return {"session_id": session_id, "messages": messages}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, claims: TokenClaims = Depends(get_current_user)):
    """Delete a single chat session and its checkpointer state."""
    from app.sessions import delete_session as _delete
    deleted = _delete(claims.sub, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.delete("/sessions")
async def delete_all_sessions(claims: TokenClaims = Depends(get_current_user)):
    """Delete all of the caller's chat sessions."""
    from app.sessions import delete_all_sessions as _delete_all
    count = _delete_all(claims.sub)
    return {"status": "deleted", "count": count}


@app.post("/sessions/archive")
async def archive_sessions(claims: TokenClaims = Depends(get_current_user)):
    """Archive all of the caller's chat sessions (soft-delete on account deletion).

    Sets ``archived_at = now()`` on every chat_sessions row owned by the
    caller and deletes the LangGraph checkpointer state for those threads.
    Archived sessions disappear from the active session list but remain in
    the table for audit (visible to admins via GET /sessions/archived).
    """
    log.info("[archive_sessions] START user=%s", claims.sub)
    from app.rag.store import get_pool
    from app.sessions import archive_user_sessions
    try:
        pool = get_pool()
        with pool.connection() as conn:
            count = archive_user_sessions(conn, claims.sub)
        log.info("[archive_sessions] SUCCESS user=%s archived=%d", claims.sub, count)
        return {"status": "archived", "count": count}
    except Exception as e:
        log.error("[archive_sessions] ERROR user=%s type=%s msg=%s",
                  claims.sub, type(e).__name__, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Archive failed: {type(e).__name__}: {e}")


@app.get("/sessions/archived")
async def list_archived_sessions(claims: TokenClaims = Depends(require_admin_user)):
    """List archived chat sessions (admin only).

    Returns sessions across all users that have been soft-archived via
    POST /sessions/archive, newest archived first.
    """
    log.info("[archived_sessions] START admin=%s", claims.sub)
    from app.sessions import list_archived_sessions as _list_archived
    sessions = _list_archived()
    log.info("[archived_sessions] SUCCESS admin=%s count=%d", claims.sub, len(sessions))
    return {"sessions": [s.to_dict() for s in sessions]}
