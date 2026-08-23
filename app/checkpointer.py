"""PostgresSaver checkpointer setup for LangGraph durability.

Paused HITL threads survive agent restarts because state is persisted
in the PostgresSaver checkpoint tables.

Note: PostgresSaver.from_conn_string() returns a context manager
(Iterator[PostgresSaver]). We enter it once and keep the instance alive
for the lifetime of the app.

The singleton connection can be closed by PostgreSQL after long idle
periods or network blips. get_checkpointer() detects closed connections
and recreates the checkpointer automatically.
"""

from __future__ import annotations

import logging

from app.config.settings import get_settings

log = logging.getLogger(__name__)

_checkpointer = None
_checkpointer_cm = None  # keep the context manager alive


def _is_connection_closed(checkpointer) -> bool:
    """Check if the PostgresSaver's underlying connection is closed."""
    try:
        conn = getattr(checkpointer, "conn", None)
        if conn is None:
            # Some versions use _conn or __conn
            conn = getattr(checkpointer, "_conn", None)
        if conn is not None:
            # psycopg3: closed attribute
            if getattr(conn, "closed", False):
                return True
            # Check if the connection is still usable
            try:
                conn.execute("SELECT 1")
            except Exception:
                return True
        return False
    except Exception:
        return True


def get_checkpointer():
    """Get or create the singleton PostgresSaver checkpointer.

    If the existing connection is closed (e.g. after a PostgreSQL restart
    or idle timeout), the checkpointer is recreated automatically.
    """
    global _checkpointer, _checkpointer_cm

    if _checkpointer is not None:
        if _is_connection_closed(_checkpointer):
            log.warning("PostgresSaver connection is closed — recreating checkpointer")
            # Clean up the old context manager
            if _checkpointer_cm is not None:
                try:
                    _checkpointer_cm.__exit__(None, None, None)
                except Exception:
                    pass
            _checkpointer = None
            _checkpointer_cm = None
        else:
            return _checkpointer

    # Create a fresh checkpointer
    from langgraph.checkpoint.postgres import PostgresSaver
    s = get_settings()
    _checkpointer_cm = PostgresSaver.from_conn_string(s.database.uri)
    _checkpointer = _checkpointer_cm.__enter__()
    _checkpointer.setup()  # creates checkpoint tables on first boot
    log.info("PostgresSaver checkpointer initialized")
    return _checkpointer


def set_checkpointer(checkpointer):
    """Replace the checkpointer (for testing)."""
    global _checkpointer
    _checkpointer = checkpointer


def reset_checkpointer():
    """Reset to None so next get_checkpointer() creates a fresh one."""
    global _checkpointer, _checkpointer_cm
    if _checkpointer_cm is not None:
        try:
            _checkpointer_cm.__exit__(None, None, None)
        except Exception:
            pass
    _checkpointer = None
    _checkpointer_cm = None
