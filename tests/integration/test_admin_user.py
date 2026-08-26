"""Integration test: admin user chat flow with HITL on write tools.

Admin tier (owner/developer super-user) → admin_ops worker → full corpus RAG
→ mock LLM plans action → HITL interrupt on write tools (trigger_scraper,
save_numbers) → /approve resumes. Read-only actions proceed without HITL.
Also tests admin token usage visibility, no budget limit, and LLM config.
"""
import json
import pytest

pytestmark = pytest.mark.integration


class TestAdminUserChat:
    def test_admin_user_gets_response(self, client, admin_headers):
        """Admin user sends a message and gets a response."""
        resp = client.post(
            "/chat",
            json={"session_id": "admin-sess-1", "message": "Show me audit logs",
                  "intent": "admin_ops"},
            headers=admin_headers,
        )
        assert resp.status_code == 200

    def test_admin_token_usage_logged(self, client, admin_headers, db_pool):
        """Admin user chat logs token usage."""
        client.post(
            "/chat",
            json={"session_id": "admin-sess-2", "message": "Show token usage",
                  "intent": "admin_ops"},
            headers=admin_headers,
        )
        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT user_sub, tier FROM agent.token_usage "
                "WHERE user_sub = 'admin-user-001' LIMIT 1"
            ).fetchone()
        assert row is not None
        assert row[0] == "admin-user-001"
        assert row[1] == "admin"

    def test_admin_hitl_on_write_tool(self, client, admin_headers, mock_llm_store):
        """Admin admin_ops flow where LLM plans trigger_scraper → HITL pause."""
        mock_llm_store.get_llm().responses = [
            "TOOL: trigger_scraper ARGS: {}"
        ]
        resp = client.post(
            "/chat",
            json={"session_id": "admin-hitl-1", "message": "Run the scraper",
                  "intent": "admin_ops"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        # trigger_scraper is a write tool → HITL pause
        assert data.get("paused") is True

    def test_admin_approve_write_tool(self, client, admin_headers, mock_llm_store):
        """Admin approves write tool → executes and writes audit log."""
        mock_llm_store.get_llm().responses = [
            "TOOL: trigger_scraper ARGS: {}"
        ]
        client.post(
            "/chat",
            json={"session_id": "admin-hitl-2", "message": "Run the scraper",
                  "intent": "admin_ops"},
            headers=admin_headers,
        )
        resp = client.post(
            "/approve",
            json={"session_id": "admin-hitl-2", "approved": True},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_admin_reject_write_tool(self, client, admin_headers, mock_llm_store):
        """Admin rejects write tool → rejection message."""
        mock_llm_store.get_llm().responses = [
            "TOOL: trigger_scraper ARGS: {}"
        ]
        client.post(
            "/chat",
            json={"session_id": "admin-hitl-3", "message": "Run the scraper",
                  "intent": "admin_ops"},
            headers=admin_headers,
        )
        resp = client.post(
            "/approve",
            json={"session_id": "admin-hitl-3", "approved": False},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "rejected" in (data.get("response") or "").lower()

    def test_admin_no_hitl_on_read_tool(self, client, admin_headers, mock_llm_store):
        """Admin admin_ops flow where LLM plans read_token_usage → no HITL."""
        mock_llm_store.get_llm().responses = [
            "TOOL: read_token_usage ARGS: {\"days\": 7}"
        ]
        resp = client.post(
            "/chat",
            json={"session_id": "admin-read-1", "message": "Show token usage",
                  "intent": "admin_ops"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Read tool → no HITL → should get a response, not paused
        assert "response" in data
        assert data.get("paused") is not True

    def test_admin_no_budget_limit(self, db_pool):
        """Admin has daily_budget_usd=0.0 → check_daily_budget always returns True."""
        from app.metering import check_daily_budget
        # Even with some token usage logged, admin should not be budget-limited.
        with db_pool.connection() as conn:
            conn.execute(
                "INSERT INTO agent.token_usage (thread_id, user_sub, tier, provider, model, "
                "prompt_tokens, completion_tokens, cost_usd, ts) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, extract(epoch from now()))",
                ("admin-user-001:test", "admin-user-001", "admin", "mock", "mock",
                 10000, 5000, 999.0),
            )
        assert check_daily_budget("admin-user-001", "admin") is True


class TestLLMConfig:
    def test_get_llm_config(self, client, free_headers):
        """Any authenticated user can read the current LLM config."""
        resp = client.get("/llm-config", headers=free_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "provider" in data
        assert "model" in data

    def test_put_llm_config_admin_only(self, client, free_headers):
        """Non-admin cannot update LLM config → 403."""
        resp = client.put(
            "/llm-config",
            json={"provider": "ollama", "model": "llama3.1:8b"},
            headers=free_headers,
        )
        assert resp.status_code == 403

    def test_put_llm_config_admin_succeeds(self, client, admin_headers, db_pool):
        """Admin can update LLM config → writes to DB + hot-reloads."""
        resp = client.put(
            "/llm-config",
            json={"provider": "ollama", "model": "llama3.1:70b",
                  "base_url": "http://ollama:11434"},
            headers=admin_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["model"] == "llama3.1:70b"

        # Verify it was written to the DB.
        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT provider, model FROM agent.llm_config ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        assert row is not None
        assert row[0] == "ollama"
        assert row[1] == "llama3.1:70b"

    def test_put_llm_config_hot_reloads(self, client, admin_headers):
        """After PUT /llm-config, the LLM store should reflect the new config."""
        from app.llm.config_store import get_llm_store

        client.put(
            "/llm-config",
            json={"provider": "mock", "model": "new-mock-model"},
            headers=admin_headers,
        )
        cfg = get_llm_store().get_config()
        assert cfg.provider == "mock"
        assert cfg.model == "new-mock-model"


class TestAdminAuditLog:
    def test_audit_log_written_for_admin_actions(self, client, admin_headers, db_pool):
        """Admin actions that trigger scraper should write to audit_log."""
        from app.security import TokenClaims
        from app.tools.admin_ops import trigger_scraper

        claims = TokenClaims(sub="admin-user-001", tier="admin", roles=[], raw_token="")
        trigger_scraper(claims)

        with db_pool.connection() as conn:
            row = conn.execute(
                "SELECT user_sub, action FROM agent.audit_log "
                "WHERE user_sub = 'admin-user-001' LIMIT 1"
            ).fetchone()
        assert row is not None
        assert row[0] == "admin-user-001"
        assert row[1] == "trigger_scraper"


class TestAdminDbTools:
    """Test the generic DB inspection tools (list_db_tables, query_db)."""

    def test_list_db_tables_agent_schema(self, db_pool):
        """list_db_tables returns tables for the agent schema."""
        from app.security import TokenClaims
        from app.tools.admin_ops import list_db_tables

        claims = TokenClaims(sub="admin-user-001", tier="admin", roles=[], raw_token="")
        tables = list_db_tables(claims, schema="agent")
        table_names = [t["table"] for t in tables]
        assert "audit_log" in table_names
        assert "chat_sessions" in table_names
        assert "token_usage" in table_names
        # Each table should have columns
        audit = [t for t in tables if t["table"] == "audit_log"][0]
        col_names = [c["column"] for c in audit["columns"]]
        assert "user_sub" in col_names
        assert "action" in col_names

    def test_query_db_select(self, db_pool):
        """query_db executes a read-only SELECT and returns rows."""
        from app.security import TokenClaims
        from app.tools.admin_ops import query_db

        claims = TokenClaims(sub="admin-user-001", tier="admin", roles=[], raw_token="")
        # Insert a test row first
        with db_pool.connection() as conn:
            conn.execute(
                "INSERT INTO agent.audit_log (user_sub, tier, action, details, ts) "
                "VALUES ('test-query-db', 'admin', 'test_action', '{}', 0)"
            )
        rows = query_db(claims, sql="SELECT user_sub, action FROM agent.audit_log WHERE user_sub = 'test-query-db'")
        assert len(rows) == 1
        assert rows[0]["user_sub"] == "test-query-db"
        assert rows[0]["action"] == "test_action"

    def test_query_db_rejects_write(self, db_pool):
        """query_db rejects non-SELECT statements."""
        from app.security import TokenClaims
        from app.tools.admin_ops import query_db
        import pytest as _pytest

        claims = TokenClaims(sub="admin-user-001", tier="admin", roles=[], raw_token="")
        with _pytest.raises(ValueError, match="Only SELECT"):
            query_db(claims, sql="INSERT INTO agent.audit_log VALUES (1)")

    def test_query_db_rejects_drop(self, db_pool):
        """query_db rejects DROP statements even with SELECT prefix."""
        from app.security import TokenClaims
        from app.tools.admin_ops import query_db
        import pytest as _pytest

        claims = TokenClaims(sub="admin-user-001", tier="admin", roles=[], raw_token="")
        with _pytest.raises(ValueError, match="drop"):
            query_db(claims, sql="SELECT 1; DROP TABLE agent.audit_log")

    def test_query_db_non_admin_rejected(self, db_pool):
        """Non-admin users cannot use query_db."""
        from app.security import TokenClaims
        from app.tools.admin_ops import query_db
        import pytest as _pytest

        claims = TokenClaims(sub="free-user-001", tier="free", roles=[], raw_token="")
        with _pytest.raises(Exception):
            query_db(claims, sql="SELECT 1")
