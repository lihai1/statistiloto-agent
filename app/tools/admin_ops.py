"""Admin operations tools — scraper trigger, audit log query.

These are gated by tier=admin in the tool allowlist AND by an explicit
JWT role check inside each function.
"""

from __future__ import annotations

import json
import logging
import time

from app.rag.store import get_pool
from app.security import TokenClaims, require_admin

log = logging.getLogger(__name__)


def trigger_scraper(claims: TokenClaims) -> dict:
    """Trigger the Go lottery scraper (admin only)."""
    require_admin(claims)
    # In production, this would call the Go service's scraper endpoint.
    # For now, log the action to the audit log.
    _audit(claims, "trigger_scraper", {})
    return {"status": "scraper_triggered"}


def query_audit_log(claims: TokenClaims, limit: int = 50) -> list[dict]:
    """Query the audit log (admin only)."""
    require_admin(claims)
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT user_sub, tier, action, details, ts FROM agent.audit_log "
            "ORDER BY ts DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return [
        {
            "user_sub": r[0],
            "tier": r[1],
            "action": r[2],
            "details": json.loads(r[3]) if isinstance(r[3], str) else r[3],
            "ts": float(r[4]),
        }
        for r in rows
    ]


def read_token_usage(claims: TokenClaims, days: int = 7) -> list[dict]:
    """Read token usage stats (admin only)."""
    require_admin(claims)
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """SELECT user_sub, tier, provider, model,
                      SUM(prompt_tokens) AS prompt_tokens,
                      SUM(completion_tokens) AS completion_tokens,
                      SUM(cost_usd) AS cost_usd,
                      COUNT(*) AS calls
               FROM agent.token_usage
               GROUP BY user_sub, tier, provider, model
               ORDER BY cost_usd DESC LIMIT 100""",
        ).fetchall()
    return [
        {
            "user_sub": r[0], "tier": r[1], "provider": r[2], "model": r[3],
            "prompt_tokens": int(r[4] or 0), "completion_tokens": int(r[5] or 0),
            "cost_usd": float(r[6] or 0), "calls": int(r[7] or 0),
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
