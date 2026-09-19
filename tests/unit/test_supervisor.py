"""Unit tests for supervisor intent classification — no DB needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.graphs.supervisor import classify_intent, route, _detect_multiple_requests, direct_tool, direct_domain


class TestClassifyIntent:
    def test_free_nl_assistant(self):
        state = {"tier": "free", "intent": "nl_assistant"}
        assert classify_intent(state) == "nl_assistant"

    def test_free_analyst_downgraded(self):
        state = {"tier": "free", "intent": "analyst"}
        assert classify_intent(state) == "nl_assistant"

    def test_free_admin_ops_downgraded(self):
        state = {"tier": "free", "intent": "admin_ops"}
        assert classify_intent(state) == "nl_assistant"

    def test_paid_nl_assistant(self):
        state = {"tier": "paid", "intent": "nl_assistant"}
        assert classify_intent(state) == "nl_assistant"

    def test_paid_analyst_allowed(self):
        state = {"tier": "paid", "intent": "analyst"}
        assert classify_intent(state) == "analyst"

    def test_paid_admin_ops_downgraded(self):
        state = {"tier": "paid", "intent": "admin_ops"}
        assert classify_intent(state) == "nl_assistant"

    def test_admin_nl_assistant(self):
        state = {"tier": "admin", "intent": "nl_assistant"}
        assert classify_intent(state) == "nl_assistant"

    def test_admin_analyst_allowed(self):
        state = {"tier": "admin", "intent": "analyst"}
        assert classify_intent(state) == "analyst"

    def test_admin_admin_ops_allowed(self):
        state = {"tier": "admin", "intent": "admin_ops"}
        assert classify_intent(state) == "admin_ops"

    def test_no_intent_defaults_to_nl_assistant(self):
        state = {"tier": "free", "intent": None}
        assert classify_intent(state) == "nl_assistant"


class TestMultiRequestDetection:
    """Tests for one-request-per-message enforcement (#15)."""

    def test_single_request_routes_normally(self):
        """A single request should NOT route to direct_multi_request."""
        state = {
            "tier": "paid",
            "user_sub": "test-user",
            "session_id": "s1",
            "message": "Generate 5 lottery forms",
        }
        result = route(state)
        assert result != "direct_multi_request"

    def test_generate_and_analyze_routes_to_multi(self):
        """'generate 5 forms and analyze 1,2,3' should route to direct_multi_request."""
        state = {
            "tier": "paid",
            "user_sub": "test-user",
            "session_id": "s2",
            "message": "generate 5 forms and analyze 1,2,3",
        }
        result = route(state)
        assert result == "direct_multi_request"

    def test_generate_and_save_exempted(self):
        """'generate 5 forms and save them' should NOT route to direct_multi_request."""
        state = {
            "tier": "paid",
            "user_sub": "test-user",
            "session_id": "s3",
            "message": "generate 5 forms and save them",
        }
        result = route(state)
        assert result != "direct_multi_request"

    def test_detect_multiple_requests_single(self):
        """_detect_multiple_requests returns empty for a single request."""
        assert _detect_multiple_requests("Generate 5 lottery forms") == []

    def test_detect_multiple_requests_two(self):
        """_detect_multiple_requests detects two operations."""
        result = _detect_multiple_requests("generate 5 forms and analyze 1,2,3")
        assert len(result) == 2

    def test_detect_multiple_requests_generate_save_exempted(self):
        """generate + save is exempted (single workflow)."""
        result = _detect_multiple_requests("generate 5 forms and save them")
        assert result == []

    def test_detect_multiple_requests_hebrew(self):
        """Hebrew conjunctions are detected."""
        result = _detect_multiple_requests("צור 5 טפסים וגם נתח 1,2,3")
        assert len(result) == 2

    def test_detect_multiple_requests_empty(self):
        """Empty message returns empty list."""
        assert _detect_multiple_requests("") == []
        assert _detect_multiple_requests("   ") == []

    def test_hot_and_cold_not_split(self):
        """Compound adjective phrase 'hot and cold' should NOT be split."""
        result = _detect_multiple_requests(
            "explain the methodology behind the hot and cold number analysis"
        )
        assert result == []

    def test_hebrew_hot_and_cold_not_split(self):
        """Hebrew compound 'חמים וקרים' should NOT be split."""
        result = _detect_multiple_requests("נתח את המספרים החמים והקרים ב-50 הגרלות האחרונות")
        assert result == []

    def test_simulate_and_analyze_still_split(self):
        """Genuine two-operation requests are still split (regression)."""
        result = _detect_multiple_requests("simulate 1,2,3,4,5,6 and analyze 7,8,9")
        assert len(result) == 2


class TestDirectToolErrorPropagation:
    """Tests for error propagation in direct_tool (#7)."""

    def test_tool_exception_returns_render_tool_error(self, monkeypatch):
        """When a tool raises an exception, direct_tool returns render_tool_error response."""
        from app.renderer import render_tool_error

        # Mock execute_tool to raise an exception.
        def _raise_exc(*args, **kwargs):
            raise RuntimeError("Service unavailable")

        monkeypatch.setattr("app.graphs.supervisor.execute_tool", _raise_exc)

        state = {
            "tier": "paid",
            "user_sub": "test-user",
            "session_id": "s1",
            "message": "Generate 5 lottery forms",
            "jwt_token": "",
        }
        result = direct_tool(state)
        # The response should be the render_tool_error text, NOT a structured result.
        assert "response" in result
        assert result["response"] == render_tool_error("en")
        # Should NOT have a tool_result with an error field.
        assert "error" not in (result.get("tool_result") or {})

    def test_tool_permission_error_returns_unauthorized(self, monkeypatch):
        """When a tool raises PermissionError, direct_tool returns unauthorized response."""
        from app.renderer import render_unauthorized

        def _raise_perm(*args, **kwargs):
            raise PermissionError("Not allowed")

        monkeypatch.setattr("app.graphs.supervisor.execute_tool", _raise_perm)

        state = {
            "tier": "free",
            "user_sub": "test-user",
            "session_id": "s2",
            "message": "Generate 5 lottery forms",
            "jwt_token": "",
        }
        result = direct_tool(state)
        assert "response" in result
        assert "not available" in result["response"].lower() or "upgrade" in result["response"].lower()


class TestDirectDomain:
    """Deterministic domain-term explanations via domain_registry (0 LLM)."""

    def _state(self, message, tier="admin"):
        return {
            "tier": tier,
            "user_sub": "test-user",
            "session_id": "s-domain",
            "message": message,
            "jwt_token": "",
        }

    def test_hebrew_probability_question_routes_direct(self):
        """'מה הסיכוי לזכות בלוטו?' → direct_domain, not the LLM."""
        assert route(self._state("מה הסיכוי לזכות בלוטו?")) == "direct_domain"

    def test_english_probability_question_routes_direct(self):
        assert route(self._state("what is the probability of winning?")) == "direct_domain"

    def test_what_is_pair_routes_direct(self):
        """Explicit domain_explanation kind with a known term → direct_domain."""
        assert route(self._state("מה זה זוג?")) == "direct_domain"

    def test_ambiguous_non_question_not_hijacked(self):
        """A statement mentioning a term without '?' must NOT route to direct_domain."""
        result = route(self._state("המספרים השמורים שלי"))
        assert result != "direct_domain"

    def test_unknown_topic_falls_through_to_llm(self):
        """A question with no registry topic still routes to a worker."""
        result = route(self._state("מה דעתך על העתיד?"))
        assert result != "direct_domain"

    def test_luck_question_routes_direct(self):
        """'מזל' is a registry topic — deterministic answer, not LLM."""
        assert route(self._state("מה דעתך על מזל בלוטו?")) == "direct_domain"

    def test_direct_domain_returns_hebrew_explanation(self):
        result = direct_domain(self._state("מה הסיכוי לזכות בלוטו?"))
        assert "16,273,488" in result["response"]
        assert "סיכוי" in result["response"]
        # disclaimer appended
        assert "נתוני עבר" in result["response"]

    def test_direct_domain_returns_english_explanation(self):
        result = direct_domain(self._state("what is the probability of winning?"))
        assert "16,273,488" in result["response"]
        assert "independent" in result["response"]

    def test_direct_domain_free_tier(self):
        """Free tier also gets the deterministic answer (no LLM needed)."""
        assert route(self._state("מה הסיכוי לזכות בלוטו?", tier="free")) == "direct_domain"


class TestSavedNumbersRouting:
    """'list/show saved numbers' must resolve to list_saved_numbers, not save_numbers."""

    def _state(self, message, tier="admin"):
        return {
            "tier": tier,
            "user_sub": "test-user",
            "session_id": "s-saved",
            "message": message,
            "jwt_token": "",
        }

    def test_list_saved_en_resolves_list_op(self):
        from app.normalizer import normalize
        from app.tool_resolver import resolve
        res = resolve(normalize("list my saved numbers"))
        assert res.operation == "list_saved_numbers"
        assert res.execution_ready

    def test_show_saved_en_resolves_list_op(self):
        from app.normalizer import normalize
        from app.tool_resolver import resolve
        res = resolve(normalize("show my saved numbers"))
        assert res.operation == "list_saved_numbers"

    def test_list_saved_he_resolves_list_op(self):
        from app.normalizer import normalize
        from app.tool_resolver import resolve
        res = resolve(normalize("הצג את המספרים השמורים שלי"))
        assert res.operation == "list_saved_numbers"

    def test_list_saved_routes_direct_tool(self):
        """Read tool, execution-ready → direct_tool (0 LLM)."""
        assert route(self._state("list my saved numbers")) == "direct_tool"

    def test_save_verb_still_resolves_save_op(self):
        from app.normalizer import normalize
        from app.tool_resolver import resolve
        res = resolve(normalize("save numbers 1,2,3,4,5,6"))
        assert res.operation == "save_numbers"
        assert route(self._state("save numbers 1,2,3,4,5,6")) == "analyst"

    def test_hebrew_save_verb_still_resolves_save_op(self):
        from app.normalizer import normalize
        from app.tool_resolver import resolve
        res = resolve(normalize("שמור את המספרים 1,2,3,4,5,6"))
        assert res.operation == "save_numbers"

    def test_saved_adjective_does_not_trigger_save(self):
        """'saved' as adjective must not match the save verb pattern."""
        from app.normalizer import normalize
        r = normalize("my saved numbers")
        assert r.request_kind == "admin_operation"  # list intent, not save


class TestSavedNumbersRenderer:
    def test_render_list_en(self):
        from app.renderer import render_structured_result
        items = [{"id": 1, "category": "default", "numbers": [1, 2, 3, 4, 5, 6]}]
        out = render_structured_result("list_saved_numbers", items, "en")
        assert "Your saved numbers (1)" in out
        assert "1 + 2 + 3 + 4 + 5 + 6" in out

    def test_render_list_he(self):
        from app.renderer import render_structured_result
        items = [{"id": 1, "category": "vip", "numbers": [7, 8, 9]}]
        out = render_structured_result("list_saved_numbers", items, "he")
        assert "המספרים השמורים שלך" in out
        assert "(vip)" in out

    def test_render_empty_he(self):
        from app.renderer import render_structured_result
        out = render_structured_result("list_saved_numbers", [], "he")
        assert "אין מספרים שמורים" in out
