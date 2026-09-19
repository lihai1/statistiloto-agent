"""Supervisor graph — top-level hierarchical router.

New architecture (Phases 2-3):
  1. route: normalize message, resolve tool, authorize, and route.
     - Trivial / out-of-scope / missing known parameter → direct_response (0 LLM)
     - Read tool, execution_ready, authorized              → direct_tool (0 LLM planner)
     - Write / ambiguous / not execution_ready             → planner workers (analyst/admin_ops/nl_assistant)

Tier gating happens via CapabilityConfig using trusted JWT claims — never mutable graph state.

SECURITY: The client-provided `intent` field is an UNTRUSTED HINT. It is logged for
audit but cannot force a route that the tier doesn't allow.

IMPORTANT: The route function returns a plain string (not Command) because conditional
edges in this LangGraph version don't support Command returns. Each direct node
re-derives the NormalizedRequest/ToolResolution from the message. This avoids a
separate normalize node, which would break interrupt propagation in subgraphs.
"""

from __future__ import annotations

import logging
import re
from typing import Literal, Optional

from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from app.graphs.nl_assistant import build_nl_assistant_graph
from app.graphs.analyst import build_analyst_graph
from app.graphs.admin_ops import build_admin_ops_graph
from app.capability import CapabilityConfig
from app.normalizer import normalize, NormalizedRequest, ConversationState
from app.normalizer import (
    FORM_GENERATION_RE,
    HEBREW_FORM_GENERATION_RE,
    ANALYZE_RE,
    HEBREW_ANALYZE_RE,
    SIMULATE_RE,
    HEBREW_SIMULATE_RE,
    STATISTICS_KEYWORDS,
    HEBREW_STATISTICS_KEYWORDS,
    ADMIN_SAVE_RE,
    HEBREW_ADMIN_SAVE_RE,
    HEBREW_RE,
)
from app.tool_resolver import resolve, ToolResolution
from app.renderer import (
    render_greeting,
    render_goodbye,
    render_capabilities,
    render_unauthorized,
    render_missing,
    render_out_of_scope,
    render_structured_result,
    render_tool_error,
    render_free_generic,
    render_multi_request,
)
from app.tool_executor import execute_tool
from app.domain_registry import detect_topic, explain
from app.free_tier_llm import is_free_llm_enabled
from app.graphs.common import append_history, emit_step

log = logging.getLogger(__name__)

# Operation verbs that indicate a distinct tool request in each segment.
# Used by _detect_multiple_requests to avoid splitting compound adjective
# phrases like "hot and cold numbers" into false multi-requests.
# Only tool-triggering verbs count — explanatory verbs (explain, what, how)
# and noun forms (analysis, statistics) are excluded.
_OPERATION_VERB_RE = re.compile(
    r"\b(generate|create|make|analyz(?:e|ed|ing|es)|analyse|"
    r"show|give|get|save|keep|store|"
    r"simulate|backtest|trigger)\b",
    re.IGNORECASE,
)
_HEBREW_OPERATION_VERB_RE = re.compile(
    r"(צור|צרי|תיצור|תייצר|נתח|נתחי|תנתח|הצג|הציגו|תן|תנו|"
    r"שמור|שמרי|תשמור|דמה|דמי|תדמה|בחן|בחני|תבחן|הפעל)",
    re.IGNORECASE,
)


class SupervisorState(TypedDict):
    user_sub: str
    tier: str                       # free | paid | admin
    session_id: str
    message: str
    intent: Optional[Literal["nl_assistant", "analyst", "admin_ops"]]
    jwt_token: str
    history: list
    context: Optional[dict]         # structured UI context (page, numbers, groupSize, etc.)
    lang: Optional[str]             # user language hint (not authoritative)
    # Client intent hint — UNTRUSTED, logged for audit, never authoritative.
    client_intent_hint: Optional[str]
    # Phase 6: conversation state for follow-up detection.
    prev_conversation_state: Optional[dict]
    conversation_state: Optional[dict]
    # Worker output bubbles up here.
    response: Optional[str]
    chunks: list
    draft: Optional[str]
    planned_tool: Optional[str]     # tool name the LLM decided to call
    tool_args: Optional[dict]       # arguments for the planned tool
    tool_result: Optional[dict]


def _authorize(tool: str | None, tier: str) -> bool:
    if not tool:
        return False
    return CapabilityConfig.is_allowed(tier, tool)


def _build_conversation_state(prev: dict | None) -> ConversationState | None:
    """Reconstruct a ConversationState from the previous checkpoint's dict."""
    if not prev or not isinstance(prev, dict):
        return None
    last_req_dict = prev.get("last_request")
    if not last_req_dict or not isinstance(last_req_dict, dict):
        return None
    try:
        last_req = NormalizedRequest(**last_req_dict)
        return ConversationState(last_request=last_req)
    except Exception:
        return None


def _derive(state: SupervisorState) -> tuple[NormalizedRequest, ToolResolution]:
    """Re-derive NormalizedRequest and ToolResolution from the message.

    Phase 6: builds ConversationState from the previous checkpoint and passes
    it to normalize() for follow-up inheritance.
    """
    conv_state = _build_conversation_state(state.get("prev_conversation_state"))
    req = normalize(
        message=state.get("message", ""),
        lang_hint=state.get("lang"),
        context=state.get("context"),
        conversation=conv_state,
        client_intent_hint=state.get("client_intent_hint"),
    )
    res = resolve(req)
    return req, res


def _detect_multiple_requests(message: str) -> list[str]:
    """Detect if a message contains multiple distinct operations.

    Splits on conjunctions (EN and HE) and checks each segment for
    operation keywords. Returns a list of detected operation descriptions
    (empty if 0 or 1 operations found).

    Exemption: if the message contains "generate" and "save" (or Hebrew
    equivalents), it's treated as a single workflow (generate-then-save)
    and NOT split.
    """
    if not message or not message.strip():
        return []

    # Exemption: generate + save = single workflow.
    has_generate = bool(FORM_GENERATION_RE.search(message) or HEBREW_FORM_GENERATION_RE.search(message))
    has_save = bool(ADMIN_SAVE_RE.search(message) or HEBREW_ADMIN_SAVE_RE.search(message))
    if has_generate and has_save:
        return []

    # Conjunctions to split on (EN + HE).
    conjunctions = [
        " and then ", " and also ", " also ", " plus ", "; ", " and ",
        " וגם ", " ואז ", " ובנוסף ",
    ]

    # Split the message on conjunctions.
    segments = [message]
    for conj in conjunctions:
        new_segments = []
        for seg in segments:
            parts = seg.split(conj)
            new_segments.extend(parts)
        segments = new_segments
    segments = [s.strip() for s in segments if s.strip()]

    if len(segments) <= 1:
        return []

    # Check each segment for operation keywords.
    operation_regexes = [
        FORM_GENERATION_RE,
        HEBREW_FORM_GENERATION_RE,
        ANALYZE_RE,
        HEBREW_ANALYZE_RE,
        SIMULATE_RE,
        HEBREW_SIMULATE_RE,
        STATISTICS_KEYWORDS,
        HEBREW_STATISTICS_KEYWORDS,
        ADMIN_SAVE_RE,
        HEBREW_ADMIN_SAVE_RE,
    ]

    detected = []
    for seg in segments:
        # Require an operation verb in the segment — this prevents compound
        # adjective phrases like "hot and cold numbers" from being split into
        # false multi-requests (both halves match STATISTICS_KEYWORDS but
        # neither contains a distinct operation verb).
        has_verb = bool(_OPERATION_VERB_RE.search(seg) or _HEBREW_OPERATION_VERB_RE.search(seg))
        if not has_verb:
            continue
        for regex in operation_regexes:
            if regex.search(seg):
                detected.append(seg)
                break  # one match per segment is enough

    return detected if len(detected) > 1 else []


def route(state: SupervisorState) -> str:
    """Normalize, resolve, authorize, and route.

    Returns a plain string route name. Each direct node re-derives the
    NormalizedRequest/ToolResolution from the message to avoid a separate
    normalize node (which would break interrupt propagation in subgraphs).
    """
    tier = state.get("tier", "free")
    user_sub = state.get("user_sub", "unknown")
    session_id = state.get("session_id", "")

    req, res = _derive(state)

    log.info("[supervisor.route] user=%s session=%s kind=%s op=%s tool=%s ready=%s",
             user_sub, session_id, req.request_kind, req.operation, res.tool, res.execution_ready)

    # Multi-request detection: if the message contains multiple distinct
    # operations, ask the user to pick one (before any other routing).
    multi = _detect_multiple_requests(state.get("message", ""))
    if len(multi) > 1:
        log.info("[supervisor.route] MULTI_REQUEST user=%s session=%s count=%d",
                 user_sub, session_id, len(multi))
        return "direct_multi_request"

    # Authorization check using CapabilityConfig (reads agent.yaml).
    if res.tool and not _authorize(res.tool, tier):
        log.info("[supervisor.route] DENY user=%s session=%s tool=%s tier=%s",
                 user_sub, session_id, res.tool, tier)
        return "direct_unauthorized"

    # Trivial → deterministic response (0 LLM).
    if req.request_kind == "trivial":
        return "direct_trivial"

    # Out-of-scope → deterministic response (0 LLM).
    if req.request_kind == "out_of_scope":
        return "direct_out_of_scope"

    # Known domain term → deterministic registry explanation (0 LLM).
    # Covers domain_explanation kind and question-form ambiguous messages
    # ("מה הסיכוי לזכות בלוטו?") that would otherwise hit the LLM. The "?"
    # guard prevents hijacking ambiguous statements that merely mention a term.
    msg = state.get("message", "")
    if req.request_kind == "domain_explanation" or (
        req.request_kind == "ambiguous" and "?" in msg
    ):
        if detect_topic(msg):
            return "direct_domain"

    # Domain explanation → NL worker (needs LLM for natural explanation).
    # Free-tier gating: when free-tier LLM is disabled (default), free users
    # get a generic deterministic response instead of calling the LLM.
    if req.request_kind == "domain_explanation":
        if tier == "free" and not is_free_llm_enabled():
            return "direct_generic"
        return "nl_assistant"

    # Missing known parameter but authorization passed → deterministic clarification.
    if not res.execution_ready and res.missing_hint:
        return "direct_clarify"

    # Execution-ready and authorized. Read tool → direct execution (0 LLM planner).
    if res.execution_ready and res.tool:
        if not CapabilityConfig.is_write_tool(res.tool):
            return "direct_tool"

    # Not ready or is a write tool → route to planner worker with pre-filled state.
    # Tier gating: free users can only use nl_assistant, paid users can use
    # nl_assistant or analyst, admin users can use all three.

    # save_numbers and list_saved_numbers are paid-tier tools handled by analyst.
    if res.tool in ("save_numbers", "list_saved_numbers"):
        if tier in ("paid", "admin"):
            return "analyst"
        return "nl_assistant"

    # Admin-only tools → admin_ops for admin, nl_assistant fallback for others.
    if req.request_kind == "admin_operation" or res.tool in (
        "query_audit_log", "read_token_usage", "search_web", "read_code",
        "edit_file", "list_files", "list_db_tables", "query_db", "trigger_scraper",
    ):
        if tier == "admin":
            return "admin_ops"
        return "nl_assistant"

    # Ambiguous or requires LLM planning — route by tier capability.
    # Free-tier gating: when free-tier LLM is disabled (default), free users
    # get a generic deterministic response instead of calling the LLM.
    if tier == "free":
        if not is_free_llm_enabled():
            return "direct_generic"
        return "nl_assistant"
    return "analyst"


def _conv_state_update(req: NormalizedRequest) -> dict:
    """Build a conversation_state update dict for the checkpointer.

    Phase 6: persists the current normalized request so the next turn
    can detect follow-ups and inherit compatible parameters.
    """
    return {
        "conversation_state": {
            "last_request": req.__dict__,
        }
    }


def direct_trivial(state: SupervisorState) -> dict:
    """Return a deterministic trivial response (greeting / capabilities / goodbye)."""
    emit_step("direct_trivial")
    req, _ = _derive(state)
    tier = state.get("tier", "free")
    lang = req.language
    trivial = req.trivial_kind

    if trivial == "greeting":
        resp = render_greeting(lang)
    elif trivial == "goodbye":
        resp = render_goodbye(lang)
    elif trivial == "capabilities":
        resp = render_capabilities(tier, lang)
    else:
        resp = render_greeting(lang)
    return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}


def direct_out_of_scope(state: SupervisorState) -> dict:
    emit_step("direct_out_of_scope")
    req, _ = _derive(state)
    resp = render_out_of_scope(req.language)
    return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}


def direct_generic(state: SupervisorState) -> dict:
    """Return a generic deterministic response for free-tier LLM-gated requests.

    Used when free-tier LLM is disabled (default) and the request is ambiguous
    or a domain explanation. Zero LLM calls.
    """
    emit_step("direct_generic")
    req, _ = _derive(state)
    resp = render_free_generic(req.language)
    return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}


def direct_domain(state: SupervisorState) -> dict:
    """Deterministic domain-term explanation from the registry (0 LLM)."""
    emit_step("direct_domain")
    req, _ = _derive(state)
    topic = detect_topic(state.get("message", ""))
    resp = explain(topic, req.language) if topic else render_free_generic(req.language)
    return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}


def direct_clarify(state: SupervisorState) -> dict:
    emit_step("direct_clarify")
    req, res = _derive(state)
    resp = render_missing(res.missing_hint, req.language)
    return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}


def direct_unauthorized(state: SupervisorState) -> dict:
    emit_step("direct_unauthorized")
    req, res = _derive(state)
    tier = state.get("tier", "free")
    resp = render_unauthorized(res.tool, tier, req.language)
    return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}


def direct_multi_request(state: SupervisorState) -> dict:
    """Ask the user to pick one request when multiple are detected."""
    emit_step("direct_multi_request")
    req, _ = _derive(state)
    lang = req.language
    requests = _detect_multiple_requests(state.get("message", ""))
    resp = render_multi_request(requests, lang)
    return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}


def direct_tool(state: SupervisorState) -> dict:
    """Execute a read tool directly (zero LLM planner calls)."""
    emit_step("direct_tool")
    req, res = _derive(state)
    jwt_token = state.get("jwt_token", "")
    lang = req.language

    if not res or not res.tool or not res.args:
        resp = render_missing(None, lang)
        return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}

    try:
        result = execute_tool(res.tool, res.args, jwt_token)
        resp = render_structured_result(res.tool, result, lang)
        return {"tool_result": result, "response": resp, "history": append_history(state, resp), **_conv_state_update(req)}
    except PermissionError as e:
        log.warning("[supervisor.direct_tool] DENIED user=%s tool=%s: %s", state.get("user_sub"), res.tool, e)
        resp = render_unauthorized(res.tool, state.get("tier", "free"), lang)
        return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}
    except Exception as e:
        log.error("[supervisor.direct_tool] ERROR user=%s tool=%s: %s", state.get("user_sub"), res.tool, e)
        resp = render_tool_error(lang)
        return {"response": resp, "history": append_history(state, resp), **_conv_state_update(req)}


def classify_intent(state: SupervisorState) -> str:
    """DEPRECATED: kept for unit-test backwards compatibility.

    Use `route()` instead; this function is a simple tier-gated fallback.
    """
    tier = state.get("tier", "free")
    intent = state.get("intent")

    if intent == "admin_ops" and tier != "admin":
        return "nl_assistant"
    if intent == "analyst" and tier == "free":
        return "nl_assistant"
    return intent or "nl_assistant"


def build_supervisor_graph(checkpointer=None):
    """Build the new supervisor graph with deterministic routing.

    The route function returns a plain string (not Command) to stay compatible
    with conditional edges. Direct nodes re-derive the normalized request from
    the message. This avoids a separate normalize node, preserving interrupt
    propagation in subgraphs (START → route → subgraph → END).
    """
    nl_graph = build_nl_assistant_graph()
    analyst_graph = build_analyst_graph()
    admin_graph = build_admin_ops_graph()

    g = StateGraph(SupervisorState)
    g.add_node("direct_trivial", direct_trivial)
    g.add_node("direct_out_of_scope", direct_out_of_scope)
    g.add_node("direct_generic", direct_generic)
    g.add_node("direct_domain", direct_domain)
    g.add_node("direct_clarify", direct_clarify)
    g.add_node("direct_unauthorized", direct_unauthorized)
    g.add_node("direct_multi_request", direct_multi_request)
    g.add_node("direct_tool", direct_tool)
    g.add_node("nl_assistant", nl_graph)
    g.add_node("analyst", analyst_graph)
    g.add_node("admin_ops", admin_graph)

    g.add_conditional_edges(START, route, {
        "direct_trivial": "direct_trivial",
        "direct_out_of_scope": "direct_out_of_scope",
        "direct_generic": "direct_generic",
        "direct_domain": "direct_domain",
        "direct_clarify": "direct_clarify",
        "direct_unauthorized": "direct_unauthorized",
        "direct_multi_request": "direct_multi_request",
        "direct_tool": "direct_tool",
        "nl_assistant": "nl_assistant",
        "analyst": "analyst",
        "admin_ops": "admin_ops",
    })
    for node in ("direct_trivial", "direct_out_of_scope", "direct_generic",
                 "direct_domain", "direct_clarify", "direct_unauthorized",
                 "direct_multi_request", "direct_tool", "nl_assistant",
                 "analyst", "admin_ops"):
        g.add_edge(node, END)

    return g.compile(checkpointer=checkpointer)
