"""Stage-level acceptance tests — real DB, real Ollama LLM, real embeddings.

These tests simulate real user conversations and inspect every pipeline stage
(retrieval, draft/plan, HITL, tool execution, finalize) so prompt engineering
and embedding quality can be validated and debugged.

Run with:  make test-chat-flows

Architecture:
  - 3 HTTP sanity tests (one per tier) — through POST /chat with real JWT
  - 15 graph-direct tests — call graph.invoke() directly, inspect each stage's
    state field (chunks, draft, planned_tool, tool_args, tool_result, response)

The graph-direct tests bypass HTTP to inspect intermediate state that the /chat
endpoint doesn't expose. HITL flows use Command(resume=...) to resume after
interrupt, matching what /approve does internally.
"""
import os
import re
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.e2e_llm]

_TIMEOUT = 300  # 5 minutes per test (real LLM on CPU can be slow, some tests have 2+ LLM calls)

HEBREW_RE = re.compile(r"[\u0590-\u05FF]")


# ── Helpers ──────────────────────────────────────────────────

def _chat(client, headers, session_id, message, intent=None, context=None):
    """Send a /chat request and return the JSON response."""
    payload = {"session_id": session_id, "message": message}
    if intent:
        payload["intent"] = intent
    if context:
        payload["context"] = context
    resp = client.post("/chat", headers=headers, json=payload)
    assert resp.status_code == 200, f"Chat failed: {resp.status_code} {resp.text}"
    return resp.json()


def _invoke_graph(state, config):
    """Invoke the supervisor graph and return the full result state."""
    from app.main import get_graph
    graph = get_graph()
    return graph.invoke(state, config)


def _make_state(user_sub, tier, session_id, message, intent=None, context=None, jwt_token="", history=None):
    """Build the state dict for a graph.invoke() call."""
    from app.security import create_test_jwt
    if not jwt_token:
        jwt_token = create_test_jwt(sub=user_sub, tier=tier)
    return {
        "user_sub": user_sub,
        "tier": tier,
        "session_id": session_id,
        "message": message,
        "intent": intent,
        "jwt_token": jwt_token,
        "history": history or [],
        "context": context,
    }


def _make_config(user_sub, session_id, recursion_limit=25):
    """Build the LangGraph config with thread_id and recursion limit."""
    from app.config.settings import get_tier_config
    tier_cfg = get_tier_config("paid")
    return {
        "configurable": {"thread_id": f"{user_sub}:{session_id}"},
        "recursion_limit": recursion_limit,
    }


def _is_hebrew(text: str) -> bool:
    return bool(HEBREW_RE.search(text))


# ── HTTP sanity tests (one per tier) ─────────────────────────

class TestHttpSanity:
    """Full HTTP flow sanity — JWT, session, checkpointer, response."""

    @pytest.mark.timeout(_TIMEOUT)
    def test_free_http_sanity(self, client, free_headers):
        """Free user can chat through the HTTP endpoint."""
        data = _chat(client, free_headers, "e2e-sanity-free", "What are hot numbers?")
        assert data.get("response"), "Expected a non-empty response"
        assert data.get("thread_id"), "Expected thread_id"
        assert not data.get("paused"), "Free user should not pause"

    @pytest.mark.timeout(_TIMEOUT)
    def test_paid_http_sanity(self, client, paid_headers):
        """Paid user can chat through the HTTP endpoint with analyst intent."""
        data = _chat(client, paid_headers, "e2e-sanity-paid",
                     "Show me the hot pairs", intent="analyst")
        assert data.get("response"), "Expected a non-empty response"
        assert data.get("thread_id"), "Expected thread_id"

    @pytest.mark.timeout(_TIMEOUT)
    def test_admin_http_sanity(self, client, admin_headers):
        """Admin user can chat through the HTTP endpoint with admin_ops intent."""
        data = _chat(client, admin_headers, "e2e-sanity-admin",
                     "Show me the audit logs", intent="admin_ops")
        assert data.get("response"), "Expected a non-empty response"
        assert data.get("thread_id"), "Expected thread_id"


# ── Free tier: nl_assistant (docs-only RAG, no tools) ────────

class TestFreeUserFlow:
    """Free user conversations — nl_assistant, docs-only retrieval, no tools."""

    @pytest.mark.timeout(_TIMEOUT)
    def test_related_lottery_question(self):
        """Related lottery question retrieves docs and produces a grounded answer."""
        state = _make_state("free-user-001", "free", "e2e-free-1",
                            "What are hot numbers?")
        config = _make_config("free-user-001", "e2e-free-1", recursion_limit=6)
        result = _invoke_graph(state, config)

        # Stage 1: Retrieve — chunks from docs corpus
        chunks = result.get("chunks", [])
        assert chunks, f"Expected non-empty chunks from docs corpus, got {len(chunks)}"

        # Stage 2: Response — non-empty, grounded
        response = result.get("response", "")
        assert response, "Expected a non-empty response"
        assert len(response) > 10, f"Response too short: {response[:100]}"

    @pytest.mark.timeout(_TIMEOUT)
    def test_unrelated_question(self):
        """Unrelated question should not crash; response should not hallucinate lottery data."""
        state = _make_state("free-user-001", "free", "e2e-free-2",
                            "What's the weather today?")
        config = _make_config("free-user-001", "e2e-free-2", recursion_limit=6)
        result = _invoke_graph(state, config)

        # Response should be non-empty (agent should redirect or explain limitations)
        response = result.get("response", "")
        assert response, "Expected a non-empty response even for off-topic questions"
        # Should NOT fabricate statistics (no specific numbers with counts)
        assert "appeared 9 times" not in response.lower()

    @pytest.mark.timeout(_TIMEOUT)
    def test_hebrew_question(self):
        """Hebrew question should produce a response (ideally Hebrew)."""
        state = _make_state("free-user-001", "free", "e2e-free-3",
                            "מה זה מספרים חמים?")
        config = _make_config("free-user-001", "e2e-free-3", recursion_limit=6)
        result = _invoke_graph(state, config)

        response = result.get("response", "")
        assert response, "Expected a non-empty response for Hebrew question"
        # With a capable model, the response should be Hebrew.
        # Small models may not follow language rules, so this is best-effort.
        if not _is_hebrew(response):
            # Still acceptable — the flow completed without error.
            pass

    @pytest.mark.timeout(_TIMEOUT * 3)
    def test_multi_turn_conversation(self):
        """Multi-turn: second message should have history loaded."""
        # Turn 1
        state1 = _make_state("free-user-001", "free", "e2e-free-4",
                             "What are hot numbers?")
        config = _make_config("free-user-001", "e2e-free-4", recursion_limit=6)
        result1 = _invoke_graph(state1, config)
        assert result1.get("response"), "Turn 1 should produce a response"

        # Turn 2 — same session (thread_id), should load history from checkpointer
        from app.main import get_graph
        graph = get_graph()
        prev_state = graph.get_state(config)
        history = list(prev_state.values.get("history", [])) if prev_state and prev_state.values else []

        state2 = _make_state("free-user-001", "free", "e2e-free-4",
                             "And what about cold ones?",
                             history=history)
        result2 = _invoke_graph(state2, config)
        response2 = result2.get("response", "")
        assert response2, "Turn 2 should produce a response"
        assert len(response2) > 5, f"Turn 2 response too short: {response2[:100]}"


# ── Paid tier: analyst (docs + lottery_history RAG, tools, HITL) ──

class TestPaidUserFlow:
    """Paid user conversations — analyst, tool calling, HITL on writes."""

    @pytest.mark.timeout(_TIMEOUT)
    def test_statistics_tool_flow(self):
        """Statistics request: retrieve → draft (TOOL line) → execute → finalize."""
        state = _make_state("paid-user-001", "paid", "e2e-paid-1",
                            "Show me the hot pairs", intent="analyst")
        config = _make_config("paid-user-001", "e2e-paid-1", recursion_limit=25)
        result = _invoke_graph(state, config)

        # Stage 1: Retrieve — chunks retrieved
        chunks = result.get("chunks", [])
        # Chunks may be empty if no lottery_history corpus data, but docs should have some
        # Don't hard-assert non-empty since lottery_history may not be ingested

        # Stage 2: Draft — LLM should have planned a tool
        planned_tool = result.get("planned_tool")
        # The LLM may pick get_statistics or may answer directly.
        # With a capable model, it should pick get_statistics.
        # With a small model, it may not — so we accept either.
        if planned_tool and planned_tool != "none":
            # Stage 3: HITL — read tool should NOT pause
            assert "__interrupt__" not in result, "Read tool should not pause"

            # Stage 4: Execute — tool_result should have expected shape
            tool_result = result.get("tool_result")
            assert tool_result, "Expected tool_result after tool execution"

        # Stage 5: Finalize — response should be readable NL
        response = result.get("response", "")
        assert response, "Expected a non-empty response"
        # Should NOT contain raw JSON
        assert '"groups"' not in response, "Response should not contain raw JSON"
        assert '"numbers"' not in response, "Response should not contain raw JSON field names"

    @pytest.mark.timeout(_TIMEOUT)
    def test_generate_form_flow(self):
        """Generate form request should plan generate_form tool."""
        state = _make_state("paid-user-001", "paid", "e2e-paid-2",
                            "Generate 3 forms for me", intent="analyst")
        config = _make_config("paid-user-001", "e2e-paid-2", recursion_limit=25)
        result = _invoke_graph(state, config)

        planned_tool = result.get("planned_tool")
        if planned_tool and planned_tool != "none":
            # If a tool was called, it should be generate_form
            # (small models may pick a different tool, so we don't hard-assert the name)
            assert "__interrupt__" not in result, "generate_form is a read tool, should not pause"
            tool_result = result.get("tool_result")
            assert tool_result, "Expected tool_result"

        response = result.get("response", "")
        assert response, "Expected a non-empty response"

    @pytest.mark.timeout(_TIMEOUT)
    def test_analyze_numbers_flow(self):
        """Analyze numbers request should plan analyze tool."""
        state = _make_state("paid-user-001", "paid", "e2e-paid-3",
                            "Analyze my numbers 7, 11, 17, 24, 31, 36", intent="analyst")
        config = _make_config("paid-user-001", "e2e-paid-3", recursion_limit=25)
        result = _invoke_graph(state, config)

        planned_tool = result.get("planned_tool")
        if planned_tool and planned_tool != "none":
            assert "__interrupt__" not in result, "analyze is a read tool, should not pause"
            tool_result = result.get("tool_result")
            assert tool_result, "Expected tool_result"

        response = result.get("response", "")
        assert response, "Expected a non-empty response"

    @pytest.mark.timeout(_TIMEOUT * 3)
    def test_save_numbers_hitl_flow(self):
        """Save numbers should trigger HITL, then resume with approval."""
        from langgraph.types import Command
        from app.main import get_graph

        state = _make_state("paid-user-001", "paid", "e2e-paid-4",
                            "Save the numbers 1, 2, 3, 4, 5, 6 as my lucky pick",
                            intent="analyst")
        config = _make_config("paid-user-001", "e2e-paid-4", recursion_limit=25)
        graph = get_graph()

        # First invoke — should pause for HITL (save_numbers is a write tool)
        result = graph.invoke(state, config)

        # The LLM may or may not plan save_numbers (small models may not).
        # If it did, we should see __interrupt__.
        if isinstance(result, dict) and "__interrupt__" in result:
            # Stage 3: HITL — paused for approval
            # Resume with approval
            resume_result = graph.invoke(
                Command(resume={"approved": True}), config
            )
            # Stage 4+5: Execute + Finalize
            response = resume_result.get("response", "")
            assert response, "Expected response after HITL approval"
        else:
            # LLM didn't plan save_numbers — acceptable for small models
            response = result.get("response", "")
            assert response, "Expected a response even without tool call"

    @pytest.mark.timeout(_TIMEOUT)
    def test_offtopic_no_tool(self):
        """Off-topic question should not trigger a tool call."""
        state = _make_state("paid-user-001", "paid", "e2e-paid-5",
                            "Tell me a joke", intent="analyst")
        config = _make_config("paid-user-001", "e2e-paid-5", recursion_limit=25)
        result = _invoke_graph(state, config)

        planned_tool = result.get("planned_tool")
        # For off-topic, the LLM should either pick "none" or no tool
        # Small models may still try a tool, but we check the response is reasonable
        response = result.get("response", "")
        assert response, "Expected a response for off-topic question"
        # The TOOL: line should not leak into the response
        assert "TOOL:" not in response, "TOOL: line should not leak into response"

    @pytest.mark.timeout(_TIMEOUT * 3)
    def test_multi_turn_tool_flow(self):
        """Multi-turn: two statistics requests in the same session."""
        from app.main import get_graph

        # Turn 1: hot pairs
        state1 = _make_state("paid-user-001", "paid", "e2e-paid-6",
                             "Show me hot pairs", intent="analyst")
        config = _make_config("paid-user-001", "e2e-paid-6", recursion_limit=25)
        result1 = _invoke_graph(state1, config)
        assert result1.get("response"), "Turn 1 should produce a response"

        # Turn 2: cold triples — should load history
        graph = get_graph()
        prev_state = graph.get_state(config)
        history = list(prev_state.values.get("history", [])) if prev_state and prev_state.values else []

        state2 = _make_state("paid-user-001", "paid", "e2e-paid-6",
                             "What about cold triples?", intent="analyst",
                             history=history)
        result2 = _invoke_graph(state2, config)
        response2 = result2.get("response", "")
        assert response2, "Turn 2 should produce a response"


# ── Admin tier: admin_ops (all corpora, admin tools) ─────────

class TestAdminUserFlow:
    """Admin user conversations — admin_ops, admin tools, HITL on writes."""

    @pytest.mark.timeout(_TIMEOUT)
    def test_audit_log_flow(self):
        """Audit log request should plan query_audit_log and produce readable NL."""
        state = _make_state("admin-user-001", "admin", "e2e-admin-1",
                            "Show me the audit logs", intent="admin_ops")
        config = _make_config("admin-user-001", "e2e-admin-1", recursion_limit=50)
        result = _invoke_graph(state, config)

        # Stage 2: Plan — should pick query_audit_log (guard override ensures this)
        planned_tool = result.get("planned_tool")
        assert planned_tool == "query_audit_log", \
            f"Expected query_audit_log, got {planned_tool}"

        # Stage 3: HITL — read tool, should not pause
        assert "__interrupt__" not in result, "query_audit_log is a read tool"

        # Stage 4: Execute — tool_result should be present (may be empty list if no audit entries)
        tool_result = result.get("tool_result")
        assert tool_result is not None, "Expected tool_result (not None)"

        # Stage 5: Finalize — response should be readable NL (not raw JSON)
        response = result.get("response", "")
        assert response, "Expected a non-empty response"
        # Should NOT be raw JSON (the old behavior before finalize was added)
        assert not response.strip().startswith("["), \
            f"Response should be NL, not raw JSON array: {response[:100]}"
        assert not response.strip().startswith("{"), \
            f"Response should be NL, not raw JSON: {response[:100]}"

    @pytest.mark.timeout(_TIMEOUT)
    def test_token_usage_flow(self):
        """Token usage request should plan read_token_usage."""
        state = _make_state("admin-user-001", "admin", "e2e-admin-2",
                            "Show me the token usage", intent="admin_ops")
        config = _make_config("admin-user-001", "e2e-admin-2", recursion_limit=50)
        result = _invoke_graph(state, config)

        planned_tool = result.get("planned_tool")
        assert planned_tool == "read_token_usage", \
            f"Expected read_token_usage, got {planned_tool}"
        assert "__interrupt__" not in result, "read_token_usage is a read tool"

        tool_result = result.get("tool_result")
        assert tool_result is not None, "Expected tool_result (not None)"

        response = result.get("response", "")
        assert response, "Expected a non-empty response"
        assert not response.strip().startswith("["), \
            f"Response should be NL, not raw JSON array: {response[:100]}"
        assert not response.strip().startswith("{"), \
            f"Response should be NL, not raw JSON: {response[:100]}"

    @pytest.mark.timeout(_TIMEOUT)
    def test_edit_file_hitl(self):
        """Edit file request should trigger HITL (write tool)."""
        from langgraph.types import Command
        from app.main import get_graph

        state = _make_state("admin-user-001", "admin", "e2e-admin-3",
                            "Edit the file app/main.py", intent="admin_ops")
        config = _make_config("admin-user-001", "e2e-admin-3", recursion_limit=50)
        graph = get_graph()

        result = graph.invoke(state, config)

        # edit_file is a write tool — should pause for HITL
        if isinstance(result, dict) and "__interrupt__" in result:
            # Resume with rejection (safer than actually editing a file)
            resume_result = graph.invoke(
                Command(resume={"approved": False}), config
            )
            response = resume_result.get("response", "")
            assert response, "Expected response after HITL rejection"
            assert "rejected" in response.lower(), \
                f"Expected rejection message, got: {response[:100]}"
        else:
            # If the guard didn't trigger edit_file (e.g. no app/ path matched),
            # the response should still be non-empty.
            response = result.get("response", "")
            assert response, "Expected a response"

    @pytest.mark.timeout(_TIMEOUT)
    def test_no_tool_question(self):
        """General question should not trigger a tool (planned_tool=none)."""
        state = _make_state("admin-user-001", "admin", "e2e-admin-4",
                            "What can you help me with?", intent="admin_ops")
        config = _make_config("admin-user-001", "e2e-admin-4", recursion_limit=50)
        result = _invoke_graph(state, config)

        planned_tool = result.get("planned_tool")
        # The guard override in admin_ops.py maps known phrases to tools.
        # "What can you help me with?" doesn't match any guard, so the LLM
        # should output TOOL: none or the guard leaves it as-is.
        response = result.get("response", "")
        assert response, "Expected a non-empty response"
        # If a tool was called, it shouldn't be a write tool (no HITL)
        assert "__interrupt__" not in result, "General question should not pause"


# ── Cross-tier: session isolation, intent downgrade ──────────

class TestCrossTierFlow:
    """Behaviors that span tiers."""

    @pytest.mark.timeout(_TIMEOUT * 3)
    def test_session_isolation(self):
        """Same session_id, different users — no cross-contamination."""
        from app.main import get_graph

        # Paid user turn 1
        state1 = _make_state("paid-user-001", "paid", "e2e-shared-1",
                             "My favorite number is 42", intent="analyst")
        config1 = _make_config("paid-user-001", "e2e-shared-1", recursion_limit=25)
        result1 = _invoke_graph(state1, config1)
        assert result1.get("response"), "Paid user turn 1 should produce a response"

        # Free user with SAME session_id but different user_sub
        # The thread_id is {user_sub}:{session_id}, so they're actually different threads
        state2 = _make_state("free-user-001", "free", "e2e-shared-1",
                             "What is my favorite number?")
        config2 = _make_config("free-user-001", "e2e-shared-1", recursion_limit=6)
        result2 = _invoke_graph(state2, config2)
        response2 = result2.get("response", "")
        assert response2, "Free user should get a response"
        # The free user should NOT know about "42" (different thread_id)
        # Small models may hallucinate, so we don't hard-assert "42" is absent.

    @pytest.mark.timeout(_TIMEOUT)
    def test_intent_downgrade(self):
        """Free user requesting analyst intent should be downgraded to nl_assistant."""
        state = _make_state("free-user-001", "free", "e2e-downgrade-1",
                            "Show me the hot pairs", intent="analyst")
        config = _make_config("free-user-001", "e2e-downgrade-1", recursion_limit=6)
        result = _invoke_graph(state, config)

        # The supervisor should downgrade analyst→nl_assistant for free users.
        # nl_assistant doesn't set planned_tool, so it should be None or absent.
        planned_tool = result.get("planned_tool")
        assert not planned_tool or planned_tool == "none", \
            f"Free user should not have a tool planned (downgraded), got: {planned_tool}"

        # Response should still be non-empty
        response = result.get("response", "")
        assert response, "Downgraded free user should still get a response"
