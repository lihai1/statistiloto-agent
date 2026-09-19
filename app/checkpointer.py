"""AsyncPostgresSaver checkpointer setup for LangGraph durability.

Paused HITL threads survive agent restarts because state is persisted
in the checkpoint tables.

The saver is backed by a psycopg ``AsyncConnectionPool`` shared by all
requests. Both are created lazily on the first event loop that calls
``get_checkpointer()`` (in tests, the session-scoped TestClient's portal
loop). ``reset_checkpointer()`` drops only the saver so tests can force
a rebuild without churning connections.
"""

from __future__ import annotations

import logging

from app.config.settings import get_settings

log = logging.getLogger(__name__)

_saver = None
_pool = None


async def get_checkpointer():
    """Get or create the singleton AsyncPostgresSaver checkpointer."""
    global _saver, _pool

    if _saver is not None:
        return _saver

    if _pool is None:
        from psycopg_pool import AsyncConnectionPool
        s = get_settings()
        _pool = AsyncConnectionPool(
            s.database.uri,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        await _pool.open()

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _saver = AsyncPostgresSaver(_pool)
    await _saver.setup()  # creates checkpoint tables on first boot
    log.info("AsyncPostgresSaver checkpointer initialized")
    return _saver


async def close_checkpointer():
    """Close the checkpointer's connection pool (app shutdown)."""
    global _saver, _pool
    _saver = None
    if _pool is not None:
        try:
            await _pool.close()
        except Exception as e:
            log.warning("Failed to close checkpointer pool: %s", e)
        _pool = None


def set_checkpointer(saver):
    """Replace the checkpointer (for testing)."""
    global _saver
    _saver = saver


def reset_checkpointer():
    """Reset the saver so next get_checkpointer() builds a fresh one.

    Keeps the connection pool — tests call this between cases and a new
    pool per test would churn connections on the portal loop.
    """
    global _saver
    _saver = None
