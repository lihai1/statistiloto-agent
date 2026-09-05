"""Redis client singleton for pub/sub streaming progress.

Provides a lazily-created ``redis.asyncio.Redis`` client used by the
``/chat/stream`` endpoint to publish incremental progress events.

The client is optional — if the ``redis`` package is not installed or
``REDIS_URL`` is not set, all helpers degrade to no-ops (returning None /
False) so the streaming endpoint falls back to inline SSE.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

_redis_client = None
_redis_checked = False


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


async def publish_event(channel: str, event: dict) -> None:
    """JSON-serialize and publish *event* to *channel*.

    Silently fails (logs a warning) when Redis is unavailable so callers
    never need to handle Redis errors.
    """
    client = get_redis()
    if client is None:
        return
    try:
        payload = json.dumps(event, default=str)
        await client.publish(channel, payload)
    except Exception as e:
        log.warning("[redis_client] Failed to publish to %s: %s", channel, e)


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
            asyncio.get_event_loop().create_task(_redis_client.aclose())
        except Exception:
            pass
    _redis_client = None
    _redis_checked = False
