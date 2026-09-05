"""Unit tests for admin_commands YAML loading — no DB needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.graphs.admin_ops import load_admin_commands


class TestLoadAdminCommands:
    """Tests for the data-driven admin command loading from YAML."""

    def test_loads_commands(self):
        """load_admin_commands returns a non-empty list of command dicts."""
        commands = load_admin_commands()
        assert len(commands) > 0

    def test_each_command_has_required_fields(self):
        """Each command has keywords, tool, args, and description."""
        commands = load_admin_commands()
        for cmd in commands:
            assert "keywords" in cmd
            assert "tool" in cmd
            assert "args" in cmd
            assert "description" in cmd
            assert isinstance(cmd["keywords"], list)
            assert len(cmd["keywords"]) > 0

    def test_token_usage_command(self):
        """The token usage command is present with correct tool."""
        commands = load_admin_commands()
        token_cmds = [c for c in commands if c["tool"] == "read_token_usage"]
        assert len(token_cmds) == 1
        assert "token" in token_cmds[0]["keywords"]
        assert token_cmds[0]["args"].get("days") == 7

    def test_audit_log_command(self):
        """The audit log command is present with correct tool."""
        commands = load_admin_commands()
        audit_cmds = [c for c in commands if c["tool"] == "query_audit_log"]
        assert len(audit_cmds) == 1
        assert "audit" in audit_cmds[0]["keywords"]
        assert audit_cmds[0]["args"].get("limit") == 50

    def test_scraper_command(self):
        """The scraper command is present with correct tool."""
        commands = load_admin_commands()
        scraper_cmds = [c for c in commands if c["tool"] == "trigger_scraper"]
        assert len(scraper_cmds) == 1
        assert "scraper" in scraper_cmds[0]["keywords"]

    def test_cached(self):
        """load_admin_commands is cached (lru_cache)."""
        c1 = load_admin_commands()
        c2 = load_admin_commands()
        assert c1 is c2  # same object (cached)
