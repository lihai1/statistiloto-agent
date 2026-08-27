"""Unit tests for CapabilityConfig — the read-only adapter over agent.yaml."""
import os
import pytest

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.config.settings import reload_settings
from app.capability import CapabilityConfig


class TestIsAllowed:
    def test_free_can_generate_form(self):
        assert CapabilityConfig.is_allowed("free", "generate_form") is True

    def test_free_can_get_statistics(self):
        assert CapabilityConfig.is_allowed("free", "get_statistics") is True

    def test_free_can_analyze(self):
        assert CapabilityConfig.is_allowed("free", "analyze") is True

    def test_free_cannot_save_numbers(self):
        assert CapabilityConfig.is_allowed("free", "save_numbers") is False

    def test_free_cannot_list_saved_numbers(self):
        assert CapabilityConfig.is_allowed("free", "list_saved_numbers") is False

    def test_free_cannot_trigger_scraper(self):
        assert CapabilityConfig.is_allowed("free", "trigger_scraper") is False

    def test_paid_can_save_numbers(self):
        assert CapabilityConfig.is_allowed("paid", "save_numbers") is True

    def test_paid_cannot_trigger_scraper(self):
        assert CapabilityConfig.is_allowed("paid", "trigger_scraper") is False

    def test_admin_can_trigger_scraper(self):
        assert CapabilityConfig.is_allowed("admin", "trigger_scraper") is True

    def test_admin_can_edit_file(self):
        assert CapabilityConfig.is_allowed("admin", "edit_file") is True

    def test_admin_can_query_db(self):
        assert CapabilityConfig.is_allowed("admin", "query_db") is True


class TestAllowedTools:
    def test_free_tools(self):
        tools = CapabilityConfig.allowed_tools("free")
        assert "generate_form" in tools
        assert "get_statistics" in tools
        assert "analyze" in tools
        assert "save_numbers" not in tools

    def test_paid_tools(self):
        tools = CapabilityConfig.allowed_tools("paid")
        assert "save_numbers" in tools
        assert "list_saved_numbers" in tools
        assert "trigger_scraper" not in tools

    def test_admin_tools(self):
        tools = CapabilityConfig.allowed_tools("admin")
        assert "trigger_scraper" in tools
        assert "edit_file" in tools
        assert "query_db" in tools


class TestWriteToolsForTier:
    def test_free_has_no_write_tools(self):
        assert CapabilityConfig.write_tools_for_tier("free") == []

    def test_paid_has_save_numbers(self):
        write = CapabilityConfig.write_tools_for_tier("paid")
        assert "save_numbers" in write
        assert "trigger_scraper" not in write

    def test_admin_has_all_write_tools(self):
        write = CapabilityConfig.write_tools_for_tier("admin")
        assert "save_numbers" in write
        assert "trigger_scraper" in write
        assert "edit_file" in write


class TestRenderUserCapabilities:
    def test_free_render_en(self):
        text = CapabilityConfig.render_user_capabilities("free", "en")
        assert "generate" in text.lower() or "statistics" in text.lower()
        assert "save" not in text.lower() or "save" not in text.lower().split("write")[0]

    def test_paid_render_en(self):
        text = CapabilityConfig.render_user_capabilities("paid", "en")
        assert "save" in text.lower()
        assert "approval" in text.lower()

    def test_admin_render_en(self):
        text = CapabilityConfig.render_user_capabilities("admin", "en")
        assert "scraper" in text.lower() or "audit" in text.lower()

    def test_free_render_he(self):
        text = CapabilityConfig.render_user_capabilities("free", "he")
        assert "היסטוריים" in text or "עבר" in text

    def test_admin_render_he(self):
        text = CapabilityConfig.render_user_capabilities("admin", "he")
        assert "עבר" in text


class TestAudienceFilter:
    """Test the audience filter helper used by the retriever."""

    def test_free_audience_is_public_only(self):
        from app.rag.retriever import _audience_filter_for_tier
        assert _audience_filter_for_tier("free") == ["public"]

    def test_paid_audience_is_public_only(self):
        from app.rag.retriever import _audience_filter_for_tier
        assert _audience_filter_for_tier("paid") == ["public"]

    def test_admin_audience_is_public_and_admin(self):
        from app.rag.retriever import _audience_filter_for_tier
        assert _audience_filter_for_tier("admin") == ["public", "admin"]


class TestAllowedCapabilitiesForTier:
    """Test the allowed capabilities helper used by the retriever."""

    def test_free_capabilities(self):
        from app.rag.retriever import _allowed_capabilities_for_tier
        caps = _allowed_capabilities_for_tier("free")
        assert "get_statistics" in caps
        assert "save_numbers" not in caps

    def test_admin_capabilities(self):
        from app.rag.retriever import _allowed_capabilities_for_tier
        caps = _allowed_capabilities_for_tier("admin")
        assert "trigger_scraper" in caps
        assert "edit_file" in caps
