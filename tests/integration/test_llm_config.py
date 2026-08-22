"""Integration test: LLM config store with DB-backed runtime config.

Tests that the config store reads from agent.llm_config, falls back to
boot defaults, and hot-reloads on DB changes.
"""
import time
import pytest

pytestmark = pytest.mark.integration


class TestLLMConfigStore:
    def test_boot_default_when_no_db_row(self, db_pool):
        """When no llm_config row exists, falls back to boot default (mock for tests)."""
        from app.llm.config_store import LLMConfigStore

        store = LLMConfigStore(poll_seconds=999, pg_pool=db_pool,
                               mock_responses=["test response"])
        cfg = store.get_config()
        assert cfg.provider == "mock"  # LLM_MOCK=true in test env
        store.stop_poller()

    def test_db_override_wins_over_boot_default(self, db_pool):
        """When a llm_config row exists, it overrides the boot default."""
        from app.llm.config_store import LLMConfigStore

        # Insert a config row.
        with db_pool.connection() as conn:
            conn.execute(
                "INSERT INTO agent.llm_config (provider, model, base_url, api_key, updated_by, updated_at) "
                "VALUES ('ollama', 'llama3.1:70b', 'http://custom:11434', '', 'admin', %s)",
                (time.time(),),
            )

        store = LLMConfigStore(poll_seconds=999, pg_pool=db_pool,
                               mock_responses=["test"])
        cfg = store.get_config()
        assert cfg.provider == "ollama"
        assert cfg.model == "llama3.1:70b"
        assert cfg.base_url == "http://custom:11434"
        store.stop_poller()

    def test_force_refresh_picks_up_new_db_row(self, db_pool):
        """force_refresh() immediately picks up a new DB config row."""
        from app.llm.config_store import LLMConfigStore

        store = LLMConfigStore(poll_seconds=999, pg_pool=db_pool,
                               mock_responses=["test"])
        initial_cfg = store.get_config()

        # Insert a new config row.
        with db_pool.connection() as conn:
            conn.execute(
                "INSERT INTO agent.llm_config (provider, model, base_url, api_key, updated_by, updated_at) "
                "VALUES ('mock', 'updated-model', '', '', 'admin', %s)",
                (time.time(),),
            )

        store.force_refresh()
        new_cfg = store.get_config()
        assert new_cfg.model == "updated-model"
        store.stop_poller()

    def test_get_llm_returns_callable(self, db_pool):
        """get_llm() returns a chat model instance that can be invoked."""
        from app.llm.config_store import LLMConfigStore, set_llm_store

        store = LLMConfigStore(poll_seconds=999, pg_pool=db_pool,
                               mock_responses=["Hello from mock LLM"])
        set_llm_store(store)
        llm = store.get_llm()
        assert llm is not None

        # FakeListChatModel returns responses in order.
        result = llm.invoke("test prompt")
        content = result.content if hasattr(result, "content") else str(result)
        assert "Hello from mock LLM" in content
        store.stop_poller()
