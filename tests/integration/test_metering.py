"""Integration test: token metering and budget enforcement.

Tests that meter_llm decorator logs to agent.token_usage and that
check_daily_budget works correctly.
"""
import time
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.integration


class TestMetering:
    def test_meter_llm_logs_token_usage(self, db_pool):
        """meter_llm decorator writes a row to agent.token_usage."""
        from app.metering import meter_llm
        from app.llm.config_store import LLMConfigStore, set_llm_store

        store = LLMConfigStore(poll_seconds=999, pg_pool=db_pool,
                               mock_responses=["test"])
        set_llm_store(store)

        @meter_llm
        def fake_llm_call(state):
            # Simulate an LLM response with usage_metadata.
            mock_resp = MagicMock()
            mock_resp.content = "test response"
            mock_resp.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
            return mock_resp

        state = {"user_sub": "meter-user", "tier": "paid", "session_id": "s1"}
        fake_llm_call(state)

        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT user_sub, tier, prompt_tokens, completion_tokens "
                "FROM agent.token_usage WHERE user_sub = 'meter-user'"
            ).fetchone()
        assert row is not None
        assert row[0] == "meter-user"
        assert row[1] == "paid"
        assert row[2] == 100
        assert row[3] == 50

        store.stop_poller()

    def test_check_daily_budget_within_limit(self, db_pool):
        """check_daily_budget returns True when under budget."""
        from app.metering import check_daily_budget

        # No usage yet → should be within budget.
        assert check_daily_budget("no-usage-user", "paid") is True

    def test_check_daily_budget_exceeded(self, db_pool):
        """check_daily_budget returns False when over budget."""
        from app.metering import check_daily_budget

        # Insert enough usage to exceed the free tier budget (0.0).
        # Free tier has daily_budget_usd=0.0, which means always True (no limit).
        # So test with paid tier (budget=5.0) and insert > 5.0 cost.
        with db_pool.connection() as conn:
            conn.execute(
                "INSERT INTO agent.token_usage "
                "(thread_id, user_sub, tier, provider, model, prompt_tokens, "
                " completion_tokens, cost_usd, ts) "
                "VALUES ('budget-user:s1', 'budget-user', 'paid', 'gemini', "
                "        'gemini-2.0-flash', 100000, 50000, 10.0, %s)",
                (time.time(),),
            )

        # Budget is 5.0, spent 10.0 → should be False.
        assert check_daily_budget("budget-user", "paid") is False

    def test_free_tier_no_budget_limit(self, db_pool):
        """Free tier has daily_budget_usd=0.0 → always within budget."""
        from app.metering import check_daily_budget

        with db_pool.connection() as conn:
            conn.execute(
                "INSERT INTO agent.token_usage "
                "(thread_id, user_sub, tier, provider, model, prompt_tokens, "
                " completion_tokens, cost_usd, ts) "
                "VALUES ('free-budget:s1', 'free-budget', 'free', 'ollama', "
                "        'llama3.1:8b', 1000, 500, 0.0, %s)",
                (time.time(),),
            )

        assert check_daily_budget("free-budget", "free") is True
