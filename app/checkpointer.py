"""PostgresSaver checkpointer setup for LangGraph durability.

Paused HITL threads survive agent restarts because state is persisted
in the PostgresSaver checkpoint tables.

Note: PostgresSaver.from_conn_string() returns a context manager
(Iterator[PostgresSaver]). We enter it once and keep the instance alive
for the lifetime of the app.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config.settings import get_settings

log = logging.getLogger(__name__)

_checkpointer = None
_checkpointer_cm = None  # keep the context manager alive


def get_checkpointer():
    """Get or create the singleton PostgresSaver checkpointer."""
    global _checkpointer, _checkpointer_cm
    if _checkpointer is not None:
        return _checkpointer

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
