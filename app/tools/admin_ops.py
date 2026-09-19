"""Admin operations tools — scraper trigger, audit log query.

These are gated by tier=admin in the tool allowlist AND by an explicit
JWT role check inside each function.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from app.rag.store import get_pool
from app.redis_client import get_sync_redis
from app.security import TokenClaims, require_admin

log = logging.getLogger(__name__)


def trigger_scraper(claims: TokenClaims) -> dict:
    """Trigger the Go lottery scraper via Redis queue (admin only)."""
    require_admin(claims)
    r = get_sync_redis()
    if r is None:
        log.warning("[admin_ops.trigger_scraper] Redis client unavailable")
        return {"status": "scraper_unavailable", "error": "redis_unavailable"}

    request_id = uuid.uuid4().hex[:12]
    now = int(time.time())
    status_key = f"scraper:status:{request_id}"
    events_key = f"scraper:events:{request_id}"

    try:
        r.xadd("scraper:requests", {
            "request_id": request_id,
            "requested_by": claims.sub,
            "source": "agent",
        }, maxlen=1000, approximate=True)

        r.hset(status_key, mapping={
            "request_id": request_id,
            "status": "queued",
            "phase": "queued",
            "requested_by": claims.sub,
            "updated_at": str(now),
        })
        r.expire(status_key, 86400)
        r.set("scraper:latest", request_id)
        _audit(claims, "trigger_scraper", {"request_id": request_id})
    except Exception as e:
        log.error("[admin_ops.trigger_scraper] Failed to enqueue request: %s", e)
        return {"status": "scraper_failed", "error": str(e)}

    # Stream writer to push progress events into SSE stream
    writer = None
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except Exception:
        pass

    last_id = "0-0"
    deadline = time.time() + 240

    while time.time() < deadline:
        try:
            entries = r.xread({events_key: last_id}, block=5000, count=100)
            if entries:
                for _, msgs in entries:
                    for msg_id, data in msgs:
                        last_id = msg_id
                        event = data.get("event")
                        phase = data.get("phase")
                        if writer and phase:
                            try:
                                writer({"event": "progress", "node": phase})
                            except Exception:
                                pass
                        if event == "done":
                            return {
                                "status": "scraper_done",
                                "inserted": int(data.get("inserted", 0)),
                                "prizes_written": int(data.get("prizes_written", 0)),
                            }
                        elif event == "error":
                            return {
                                "status": "scraper_failed",
                                "error": data.get("message", "Scraper execution failed"),
                            }
        except Exception as e:
            log.warning("[admin_ops.trigger_scraper] XREAD warning: %s", e)

        # Check status hash fallback on slice timeout
        try:
            st = r.hgetall(status_key)
            if st:
                status = st.get("status")
                if status == "done":
                    return {
                        "status": "scraper_done",
                        "inserted": int(st.get("inserted", 0)),
                        "prizes_written": int(st.get("prizes_written", 0)),
                    }
                elif status in ("error", "interrupted"):
                    return {
                        "status": "scraper_failed",
                        "error": st.get("error", "Scraper execution failed"),
                    }
        except Exception as e:
            log.warning("[admin_ops.trigger_scraper] HGETALL warning: %s", e)

    return {"status": "scraper_running", "request_id": request_id}


def _keycloak_user_join_available(pool) -> bool:
    """Check whether ``keycloak.user_entity`` exists in this database.

    The agent's own dev/test DB (docker-compose-dev.yml) only provisions the
    ``agent`` schema, so the join is skipped there and ``user_sub`` is
    returned as-is. In the full stack the Keycloak schema is present and the
    join resolves ``sub`` → email.
    """
    try:
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT to_regclass('keycloak.user_entity')"
            ).fetchone()
        return row is not None and row[0] is not None
    except Exception:
        return False


def query_audit_log(claims: TokenClaims, limit: int = 50) -> list[dict]:
    """Query the audit log (admin only).

    When the Keycloak schema is available, joins ``agent.audit_log`` with
    ``keycloak.user_entity`` so the UI can display the user's email instead
    of the opaque JWT ``sub``. Otherwise falls back to ``user_sub``.
    """
    require_admin(claims)
    pool = get_pool()
    join = _keycloak_user_join_available(pool)
    if join:
        sql = (
            "SELECT a.user_sub, COALESCE(u.email, u.username, a.user_sub) AS user_email, "
            "a.tier, a.action, a.details, a.ts "
            "FROM agent.audit_log a "
            "LEFT JOIN keycloak.user_entity u ON u.id = a.user_sub "
            "ORDER BY a.ts DESC LIMIT %s"
        )
    else:
        sql = (
            "SELECT user_sub, user_sub AS user_email, tier, action, details, ts "
            "FROM agent.audit_log ORDER BY ts DESC LIMIT %s"
        )
    with pool.connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [
        {
            "user_sub": r[0],
            "user_email": r[1],
            "tier": r[2],
            "action": r[3],
            "details": json.loads(r[4]) if isinstance(r[4], str) else r[4],
            "ts": float(r[5]),
        }
        for r in rows
    ]


def read_token_usage(claims: TokenClaims, days: int = 7) -> list[dict]:
    """Read token usage stats (admin only).

    When the Keycloak schema is available, joins ``agent.token_usage`` with
    ``keycloak.user_entity`` so the UI can display the user's email instead
    of the opaque JWT ``sub``. Otherwise falls back to ``user_sub``.
    """
    require_admin(claims)
    pool = get_pool()
    join = _keycloak_user_join_available(pool)
    if join:
        sql = (
            "SELECT t.user_sub, COALESCE(u.email, u.username, t.user_sub) AS user_email, "
            "t.tier, t.provider, t.model, "
            "SUM(t.prompt_tokens) AS prompt_tokens, "
            "SUM(t.completion_tokens) AS completion_tokens, "
            "SUM(t.cost_usd) AS cost_usd, "
            "COUNT(*) AS calls "
            "FROM agent.token_usage t "
            "LEFT JOIN keycloak.user_entity u ON u.id = t.user_sub "
            "GROUP BY t.user_sub, u.email, u.username, t.tier, t.provider, t.model "
            "ORDER BY cost_usd DESC LIMIT 100"
        )
    else:
        sql = (
            "SELECT user_sub, user_sub AS user_email, tier, provider, model, "
            "SUM(prompt_tokens) AS prompt_tokens, "
            "SUM(completion_tokens) AS completion_tokens, "
            "SUM(cost_usd) AS cost_usd, "
            "COUNT(*) AS calls "
            "FROM agent.token_usage "
            "GROUP BY user_sub, tier, provider, model "
            "ORDER BY cost_usd DESC LIMIT 100"
        )
    with pool.connection() as conn:
        rows = conn.execute(sql).fetchall()
    return [
        {
            "user_sub": r[0], "user_email": r[1], "tier": r[2],
            "provider": r[3], "model": r[4],
            "prompt_tokens": int(r[5] or 0), "completion_tokens": int(r[6] or 0),
            "cost_usd": float(r[7] or 0), "calls": int(r[8] or 0),
        }
        for r in rows
    ]


def _audit(claims: TokenClaims, action: str, details: dict):
    """Write to the audit log."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO agent.audit_log (user_sub, tier, action, details, ts)
               VALUES (%s, %s, %s, %s, %s)""",
            (claims.sub, claims.tier, action, json.dumps(details), time.time()),
        )


# ── Generic DB inspection tools (admin only) ─────────────────

# READ-ONLY guard: only allow SELECT statements. No INSERT/UPDATE/DELETE/
# DROP/ALTER/TRUNCATE/etc. The admin can view any data but cannot modify
# the DB through this tool — write operations go through dedicated tools
# with HITL approval.
_FORBIDDEN_KEYWORDS = frozenset({
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "grant", "revoke", "vacuum", "copy", "merge",
})

# Functions with side effects or abuse potential — allowed by SELECT-only
# keyword checks but can still write state or stall the pool.
_FORBIDDEN_FUNCTIONS_RE = re.compile(
    r"\b(pg_sleep|set_config|nextval|setval|currval|pg_advisory_\w+|"
    r"dblink\w*|pg_terminate_backend|pg_cancel_backend|lo_\w+|"
    r"pg_notification_queue_usage|pg_notify|txid_\w*|pg_export_snapshot)\s*\(",
    re.IGNORECASE,
)


def _validate_readonly_sql(sql: str) -> None:
    """Raise ValueError if the SQL is not a read-only SELECT statement."""
    stripped = sql.strip().lower().rstrip(";").strip()
    if not stripped.startswith("select") and not stripped.startswith("with"):
        raise ValueError("Only SELECT or WITH (CTE) queries are allowed.")
    # Check for forbidden keywords as whole words (not inside strings/identifiers).
    # Simple heuristic: tokenize on whitespace and check.
    tokens = re.findall(r"\b\w+\b", stripped)
    for token in tokens:
        if token in _FORBIDDEN_KEYWORDS:
            raise ValueError(f"Keyword '{token}' is not allowed in read-only queries.")
    if _FORBIDDEN_FUNCTIONS_RE.search(stripped):
        raise ValueError("Query contains a function with side effects; not allowed.")


def list_db_tables(claims: TokenClaims, schema: str = "agent") -> list[dict]:
    """List all tables in a database schema (admin only).

    Args:
        schema: the schema name (e.g. 'agent', 'lottery', 'public').
    """
    require_admin(claims)
    _audit(claims, "list_db_tables", {"schema": schema})
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT table_name, column_name, data_type
               FROM information_schema.columns
               WHERE table_schema = %s
               ORDER BY table_name, ordinal_position""",
            (schema,),
        ).fetchall()
    # Group columns by table.
    tables: dict[str, list[dict]] = {}
    for r in rows:
        table_name = r[0]
        if table_name not in tables:
            tables[table_name] = []
        tables[table_name].append({"column": r[1], "type": r[2]})
    return [
        {"table": t, "columns": cols}
        for t, cols in tables.items()
    ]


def query_db(claims: TokenClaims, sql: str, limit: int = 50) -> list[dict]:
    """Execute a read-only SQL query against the database (admin only).

    Only SELECT or WITH (CTE) queries are allowed. The result is limited
    to ``limit`` rows (default 50, max 200).

    Args:
        sql: a read-only SQL SELECT query.
        limit: max number of rows to return (default 50, max 200).
    """
    require_admin(claims)
    _validate_readonly_sql(sql)
    limit = min(max(limit, 1), 200)  # clamp 1–200
    _audit(claims, "query_db", {"sql": sql[:200], "limit": limit})
    pool = get_pool()
    with pool.connection() as conn:
        # Defense-in-depth: READ ONLY transaction + statement timeout so a
        # side-effect function or runaway query can't write state or stall.
        conn.execute("BEGIN READ ONLY")
        try:
            conn.execute("SET LOCAL statement_timeout = '10s'")
            # Wrap in a subquery to enforce the LIMIT safely.
            rows = conn.execute(f"SELECT * FROM ({sql.rstrip(';')}) AS _q LIMIT %s", (limit,))
            col_names = [desc[0] for desc in rows.description]
            result = []
            for row in rows.fetchall():
                result.append(dict(zip(col_names, row)))
        finally:
            conn.execute("ROLLBACK")
    return result
