"""Unit tests for the free-tier LLM toggle and config — no DB needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.config.settings import get_settings, reload_settings
from app.free_tier_llm import (
    is_free_llm_enabled,
    set_free_llm_enabled,
    reset_free_llm_toggle,
)
from app.renderer import render_free_generic


class TestFreeTierConfig:
    def test_default_disabled(self):
        """free_tier.llm_enabled defaults to False."""
        reload_settings()
        reset_free_llm_toggle()
        s = get_settings()
        assert s.free_tier.llm_enabled is False

    def test_env_override_enables(self):
        """FREE_LLM_ENABLED=true sets the config to True."""
        os.environ["FREE_LLM_ENABLED"] = "true"
        reload_settings()
        reset_free_llm_toggle()
        try:
            s = get_settings()
            assert s.free_tier.llm_enabled is True
            assert is_free_llm_enabled() is True
        finally:
            del os.environ["FREE_LLM_ENABLED"]
            reload_settings()
            reset_free_llm_toggle()

    def test_env_override_disables(self):
        """FREE_LLM_ENABLED=false explicitly disables."""
        os.environ["FREE_LLM_ENABLED"] = "false"
        reload_settings()
        reset_free_llm_toggle()
        try:
            assert is_free_llm_enabled() is False
        finally:
            del os.environ["FREE_LLM_ENABLED"]
            reload_settings()
            reset_free_llm_toggle()


class TestFreeTierToggle:
    def test_toggle_set_and_reset(self):
        """set_free_llm_enabled changes the state; reset clears it."""
        reset_free_llm_toggle()
        assert is_free_llm_enabled() is False

        set_free_llm_enabled(True)
        assert is_free_llm_enabled() is True

        set_free_llm_enabled(False)
        assert is_free_llm_enabled() is False

        reset_free_llm_toggle()

    def test_toggle_reads_settings_on_first_call(self):
        """The toggle lazily reads from settings on first access."""
        reset_free_llm_toggle()
        reload_settings()
        # Default config has llm_enabled=False
        assert is_free_llm_enabled() is False
        reset_free_llm_toggle()


class TestRenderFreeGeneric:
    def test_english_response(self):
        resp = render_free_generic("en")
        assert "generate forms" in resp.lower()
        assert "statistics" in resp.lower()
        assert "paid plan" in resp.lower()

    def test_hebrew_response(self):
        resp = render_free_generic("he")
        assert "חינם" in resp or "בתשלום" in resp

    def test_default_language_is_english(self):
        resp = render_free_generic()
        assert "generate forms" in resp.lower()
