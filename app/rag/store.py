"""PostgreSQL connection pool for the agent schema (pgvector + metering).

Uses psycopg3 ConnectionPool. The pool is initialized lazily on first access
so tests can inject a mock pool before any RAG/metering code runs.
"""

from __future__ import annotations

import logging
from typing import Optional

from psycopg_pool import ConnectionPool

from app.config.settings import get_settings

log = logging.getLogger(__name__)

_pool: Optional[ConnectionPool] = None


def get_pool() -> ConnectionPool:
    """Get or create the singleton connection pool."""
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = ConnectionPool(s.database.uri, min_size=1, max_size=10, open=True)
        log.info("PostgreSQL pool created: %s", _mask_uri(s.database.uri))
    return _pool


def set_pool(pool: ConnectionPool | None, close_old: bool = False):
    """Replace the pool (for testing).

    Args:
        pool: the new pool to use, or None to clear.
        close_old: if True, close the old pool before replacing.
            Defaults to False so session-scoped test pools aren't closed
            prematurely by function-scoped fixtures.
    """
    global _pool
    if close_old and _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
    _pool = pool


def _mask_uri(uri: str) -> str:
    """Mask password in URI for logging."""
    if "@" in uri:
        prefix, suffix = uri.split("@", 1)
        if "://" in prefix:
            scheme, rest = prefix.split("://", 1)
            if ":" in rest:
                user, _pw = rest.split(":", 1)
                return f"{scheme}://{user}:***@{suffix}"
    return uri


# Alias for the plan's naming convention
pg_pool = None  # set to get_pool() on first access via property below


class _PoolProxy:
    """Lazy proxy that delegates to get_pool() — allows `from app.rag.store import pg_pool`."""
    def __getattr__(self, name):
        return getattr(get_pool(), name)

    def __enter__(self):
        return get_pool().__enter__()

    def __exit__(self, *args):
        return get_pool().__exit__(*args)


pg_pool = _PoolProxy()
