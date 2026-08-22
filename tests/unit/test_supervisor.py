"""Unit tests for supervisor intent classification — no DB needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.graphs.supervisor import classify_intent


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
