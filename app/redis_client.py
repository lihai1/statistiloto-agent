"""Redis client singleton for Streams-based streaming progress.

Provides a lazily-created ``redis.asyncio.Redis`` client used by the
``/chat/stream`` endpoint to publish incremental progress events to a
Redis Stream (``agent:stream:{thread_id}:{run_id}``).

Each run gets its own stream key: a thread (session) can produce many
runs — the first message, every follow-up message, and each HITL resume
— and a replay of a previous run's events would surface stale terminal
events (``done``/``paused``) to the relay before the new run's events.

Streams are replayable — unlike pub/sub, a subscriber that connects late
(or after the run already finished on a fast deterministic path) can read
the full event history with ``XRANGE``/``XREAD`` from ``0-0``. Keys expire
after one hour.

The client is optional — if the ``redis`` package is not installed or
``REDIS_URL`` is not set, all helpers degrade to no-ops (returning None /
False) so the streaming endpoint falls back to inline SSE.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Optional

log = logging.getLogger(__name__)

_redis_client = None
_redis_checked = False
_sync_redis_client = None
_sync_redis_checked = False

STREAM_TTL_SECONDS = 3600
STREAM_MAXLEN = 2000


def stream_key(thread_id: str) -> str:
    """Return a fresh Redis Stream key for one run's events.

    A unique run suffix isolates each run's stream: replaying with
    ``XREAD`` from ``0-0`` then yields only this run's events, never a
    previous run's terminal ``done``/``paused``.
    """
    return f"agent:stream:{thread_id}:{uuid.uuid4().hex[:12]}"


def get_redis():
    """Return a singleton ``redis.asyncio.Redis`` client, or None.

    Returns None when:
      - the ``redis`` package is not installed, or
      - ``REDIS_URL`` is not set in the environment.

    The client is created once and reused for the lifetime of the process.
    """
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True

    url = os.environ.get("REDIS_URL")
    if not url:
        log.debug("[redis_client] REDIS_URL not set — Redis disabled")
        return None

    try:
        import redis.asyncio as aioredis  # type: ignore
    except ImportError:
        log.warning("[redis_client] redis package not installed — Redis disabled")
        return None

    try:
        _redis_client = aioredis.from_url(url, decode_responses=True)
        log.info("[redis_client] Redis client created for %s", url)
    except Exception as e:
        log.warning("[redis_client] Failed to create Redis client: %s", e)
        _redis_client = None
    return _redis_client


def get_sync_redis():
    """Return a singleton synchronous ``redis.Redis`` client, or None."""
    global _sync_redis_client, _sync_redis_checked
    if _sync_redis_checked:
        return _sync_redis_client
    _sync_redis_checked = True

    url = os.environ.get("REDIS_URL")
    if not url:
        return None

    try:
        import redis
    except ImportError:
        log.warning("[redis_client] redis package not installed")
        return None

    try:
        _sync_redis_client = redis.from_url(url, decode_responses=True)
        log.info("[redis_client] Sync Redis client created for %s", url)
    except Exception as e:
        log.warning("[redis_client] Failed to create sync Redis client: %s", e)
        _sync_redis_client = None
    return _sync_redis_client


async def publish_event(key: str, event: dict) -> None:
    """JSON-serialize and append *event* to the Redis Stream *key*.

    Uses XADD with MAXLEN ~2000 (approximate) and refreshes the key TTL.
    Silently fails (logs a warning) when Redis is unavailable so callers
    never need to handle Redis errors.
    """
    client = get_redis()
    if client is None:
        return
    try:
        payload = json.dumps(event, default=str)
        await client.xadd(key, {"data": payload}, maxlen=STREAM_MAXLEN, approximate=True)
        await client.expire(key, STREAM_TTL_SECONDS)
    except Exception as e:
        log.warning("[redis_client] Failed to XADD to %s: %s", key, e)


async def is_redis_available() -> bool:
    """Check if the Redis client is available and connected."""
    client = get_redis()
    if client is None:
        return False
    try:
        await client.ping()
        return True
    except Exception:
        return False


def reset_redis_client() -> None:
    """Reset the singleton (for testing)."""
    global _redis_client, _redis_checked
    if _redis_client is not None:
        try:
            import asyncio
            asyncio.get_running_loop().create_task(_redis_client.aclose())
        except RuntimeError:
            try:
                asyncio.run(_redis_client.aclose())
            except Exception:
                pass
        except Exception:
            pass
    _redis_client = None
    _redis_checked = False
