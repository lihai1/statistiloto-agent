"""ToolExecutor — defense-in-depth tool dispatch.

Re-authorizes every tool call using the JWT token's claims (tier, sub) instead of
mutable graph state. Fails closed for unauthorized or unknown tools.

The executor does NOT decide whether HITL is needed — the LangGraph HITL gate
handles that before dispatching here.
"""

from __future__ import annotations

import logging
from typing import Any

from app.capability import CapabilityConfig
from app.security import TokenClaims, validate_jwt
from app.tools import lottery_grpc, saved_numbers

log = logging.getLogger(__name__)


def _reconstruct_claims(jwt_token: str) -> TokenClaims:
    """Validate the raw JWT and return trusted claims.

    Re-authorization uses the JWT token itself, not graph state, to prevent
    tampering. In dev/test mode (JWT_VERIFY=false), the signature is skipped.
    """
    # validate_jwt expects "Bearer <token>" format.
    return validate_jwt(f"Bearer {jwt_token}")


def _authorize_tool(tool_name: str, claims: TokenClaims) -> None:
    """Re-authorize the tool against JWT claims. Fails closed."""
    if not CapabilityConfig.is_allowed(claims.tier, tool_name):
        log.warning("[tool_executor] DENIED user=%s tier=%s tool=%s", claims.sub, claims.tier, tool_name)
        raise PermissionError(
            f"Tool '{tool_name}' is not authorized for tier '{claims.tier}'. "
            f"This request has been denied and logged."
        )


def execute_tool(tool_name: str, args: dict, jwt_token: str) -> dict:
    """Execute a tool by name with defense-in-depth JWT re-authorization.

    Args:
        tool_name: the tool to execute.
        args: the tool arguments (already validated by ToolResolver).
        jwt_token: the user's raw JWT token.

    Returns:
        The tool result, or an error dict if execution failed.

    Raises:
        PermissionError: if the tool is not allowed for the user's tier.
        ValueError: if the tool is unknown.
    """
    claims = _reconstruct_claims(jwt_token)
    log.info("[tool_executor] START user=%s tier=%s tool=%s", claims.sub, claims.tier, tool_name)

    # Re-authorize against trusted JWT claims (defense-in-depth).
    _authorize_tool(tool_name, claims)

    # Dispatch
    try:
        result = _dispatch(tool_name, args, claims)
    except Exception as e:
        log.error("[tool_executor] ERROR user=%s tool=%s msg=%s", claims.sub, tool_name, e, exc_info=True)
        return {"error": str(e), "tool": tool_name}

    log.info("[tool_executor] SUCCESS user=%s tool=%s", claims.sub, tool_name)
    return result


def _dispatch(tool_name: str, args: dict, claims: TokenClaims) -> Any:
    """Dispatch to the real tool implementation."""
    if tool_name == "generate_form":
        return lottery_grpc.generate_form(
            how_many=args.get("how_many", 1),
            form_type=args.get("form_type", 6),
            will_be=args.get("will_be"),
            strength=args.get("strength", 2),
            window_from=args.get("window_from"),
            window_to=args.get("window_to"),
        )
    if tool_name == "get_statistics":
        return lottery_grpc.get_statistics(
            how_many=args.get("how_many", 10),
            group_size=args.get("group_size", args.get("form_type", 2)),
            strength=args.get("strength", "hot"),
            window_from=args.get("window_from"),
            window_to=args.get("window_to"),
        )
    if tool_name == "analyze":
        return lottery_grpc.analyze(
            form=args.get("form", []),
            window_from=args.get("window_from"),
            window_to=args.get("window_to"),
        )
    if tool_name == "list_saved_numbers":
        return saved_numbers.list_saved_numbers(user_sub=claims.sub, jwt_token=claims.raw_token)
    if tool_name == "save_numbers":
        return saved_numbers.save_numbers(
            user_sub=claims.sub,
            jwt_token=claims.raw_token,
            category=args.get("category", "default"),
            numbers=args.get("numbers", []),
            will_be=args.get("will_be"),
        )
    if tool_name == "query_audit_log":
        from app.tools import admin_ops
        return admin_ops.query_audit_log(claims=claims, limit=args.get("limit", 50))
    if tool_name == "read_token_usage":
        from app.tools import admin_ops
        return admin_ops.read_token_usage(claims=claims, days=args.get("days", 7))
    if tool_name == "trigger_scraper":
        from app.tools import admin_ops
        return admin_ops.trigger_scraper(claims=claims)
    if tool_name == "list_db_tables":
        from app.tools import admin_ops
        return admin_ops.list_db_tables(claims=claims, schema=args.get("schema", "agent"))
    if tool_name == "query_db":
        from app.tools import admin_ops
        return admin_ops.query_db(claims=claims, sql=args.get("sql", ""), limit=args.get("limit", 50))
    # Tools not yet implemented in admin_ops.py — return a clear error.
    if tool_name in ("edit_file", "read_code", "list_files", "search_web"):
        return {"error": f"Tool '{tool_name}' is not yet implemented."}

    raise ValueError(f"Unknown tool: {tool_name}")
