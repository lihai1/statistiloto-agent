"""Unit tests for config settings — no DB needed."""

import os
import pytest

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.config.settings import get_settings, get_tier_config, reload_settings


class TestSettings:
    def test_loads_yaml_defaults(self):
        reload_settings()
        s = get_settings()
        assert s.llm.provider == "ollama"
        assert s.llm.ollama.model == "qwen3:8b"

    def test_env_overrides(self):
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_MODEL"] = "gemini-2.5-pro"
        reload_settings()
        s = get_settings()
        assert s.llm.provider == "gemini"
        assert s.llm.gemini.model == "gemini-2.5-pro"
        # Clean up
        del os.environ["LLM_PROVIDER"]
        del os.environ["GEMINI_MODEL"]
        reload_settings()

    def test_tier_configs_loaded(self):
        reload_settings()
        s = get_settings()
        assert "free" in s.tiers
        assert "paid" in s.tiers
        assert "admin" in s.tiers

    def test_free_tier_caps(self):
        reload_settings()
        cfg = get_tier_config("free")
        assert cfg.recursion_limit == 6
        assert "generate_form" in cfg.allowed_tools
        assert cfg.daily_budget_usd == 0.0

    def test_paid_tier_caps(self):
        reload_settings()
        cfg = get_tier_config("paid")
        assert cfg.recursion_limit == 25
        assert "save_numbers" in cfg.allowed_tools
        assert "lottery_history" in cfg.rag_corpora
        assert cfg.daily_budget_usd == 5.0

    def test_admin_tier_caps(self):
        reload_settings()
        cfg = get_tier_config("admin")
        assert cfg.recursion_limit == 50
        assert "trigger_scraper" in cfg.allowed_tools
        assert "ops_logs" in cfg.rag_corpora
        assert "user_data" in cfg.rag_corpora
        # Admin is the owner — no budget limit
        assert cfg.daily_budget_usd == 0.0

    def test_admin_has_all_write_tools(self):
        """Admin can call write tools (save_numbers, trigger_scraper)."""
        reload_settings()
        cfg = get_tier_config("admin")
        assert "save_numbers" in cfg.allowed_tools
        assert "trigger_scraper" in cfg.allowed_tools

    def test_paid_has_write_tool(self):
        """Paid tier can call save_numbers (write tool) — HITL will trigger."""
        reload_settings()
        cfg = get_tier_config("paid")
        assert "save_numbers" in cfg.allowed_tools

    def test_free_has_no_write_tools(self):
        """Free tier has no write tools — HITL will never trigger for free."""
        reload_settings()
        cfg = get_tier_config("free")
        from app.tools.registry import WRITE_TOOLS
        for tool in cfg.allowed_tools:
            assert tool not in WRITE_TOOLS

    def test_unknown_tier_falls_back_to_free(self):
        reload_settings()
        cfg = get_tier_config("nonexistent")
        assert cfg.recursion_limit == 6  # free default
