"""Unit tests for supervisor intent classification — no DB needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.graphs.supervisor import classify_intent, route, _detect_multiple_requests, direct_tool


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
