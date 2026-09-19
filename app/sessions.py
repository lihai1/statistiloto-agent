"""Chat session history — list, load, delete user conversations.

Sessions are indexed in agent.chat_sessions (title, preview, timestamps).
The full message history is reconstructed from the LangGraph checkpointer
state (the `history` channel), so we never duplicate messages here.

Tier-based retention limits:
  free   → 1 saved session
  paid   → 15 saved sessions
  admin  → unlimited

When a user exceeds their limit, the oldest sessions (by updated_at) are
pruned automatically — both the chat_sessions row and the checkpointer
state for that thread are deleted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from app.rag.store import get_pool

log = logging.getLogger(__name__)

# Tier → max saved sessions. None means unlimited.
TIER_SESSION_LIMITS: dict[str, Optional[int]] = {
    "free": 1,
    "paid": 15,
    "admin": None,
}


@dataclass
class ChatSession:
    session_id: str
    thread_id: str
    title: str
    last_message: str
    message_count: int
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "title": self.title,
            "last_message": self.last_message,
            "message_count": self.message_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _tier_limit(tier: str) -> Optional[int]:
    return TIER_SESSION_LIMITS.get(tier, TIER_SESSION_LIMITS["free"])


def _thread_id(user_sub: str, session_id: str) -> str:
    return f"{user_sub}:{session_id}"


def upsert_session(
    user_sub: str,
    session_id: str,
    title: str,
    last_message: str,
    tier: str,
) -> None:
    """Insert or update a chat session row.

    Called after each chat turn. The title is set from the first user
    message and left untouched on subsequent turns. message_count is
    incremented. After upserting, the tier limit is enforced by pruning
    the oldest sessions if the user is over quota.
    """
    pool = get_pool()
    thread_id = _thread_id(user_sub, session_id)
    now = time.time()
    with pool.connection() as conn:
        # Upsert — only set title on first insert (when row doesn't exist).
        conn.execute(
            """
            INSERT INTO agent.chat_sessions
                (user_sub, session_id, thread_id, title, last_message, message_count, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, 1, %s, %s)
            ON CONFLICT (user_sub, session_id) DO UPDATE
            SET last_message = EXCLUDED.last_message,
                message_count = agent.chat_sessions.message_count + 1,
                updated_at = EXCLUDED.updated_at
            """,
            (user_sub, session_id, thread_id, title, last_message, now, now),
        )
    _enforce_limit(user_sub, tier)


def _enforce_limit(user_sub: str, tier: str) -> None:
    """Delete oldest sessions (and their checkpointer state) if over the tier limit."""
    limit = _tier_limit(tier)
    if limit is None:
        return  # unlimited

    pool = get_pool()
    with pool.connection() as conn:
        # Count this user's sessions.
        row = conn.execute(
            "SELECT COUNT(*) FROM agent.chat_sessions WHERE user_sub = %s",
            (user_sub,),
        ).fetchone()
        count = row[0] if row else 0

        if count <= limit:
            return

        # Select the oldest sessions to delete (by updated_at ASC).
        to_delete = count - limit
        rows = conn.execute(
            """
            DELETE FROM agent.chat_sessions
            WHERE id IN (
                SELECT id FROM agent.chat_sessions
                WHERE user_sub = %s
                ORDER BY updated_at ASC
                LIMIT %s
            )
            RETURNING thread_id
            """,
            (user_sub, to_delete),
        ).fetchall()

        # Also clean up the checkpointer state for those threads.
        for r in rows:
            thread_id = r[0]
            _delete_checkpointer_state(conn, thread_id)
        log.info("[sessions] Pruned %d old session(s) for user=%s tier=%s", len(rows), user_sub, tier)


def _delete_checkpointer_state(conn, thread_id: str) -> None:
    """Delete all checkpointer rows for a thread.

    The PostgresSaver creates tables `checkpoints`, `checkpoint_blobs`, and
    `checkpoint_writes` in the public schema (or whatever search_path is active).
    We delete by thread_id which is safe — each thread belongs to one user.
    """
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        try:
            conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (thread_id,))
        except Exception as e:
            log.debug("[sessions] Failed to delete from %s for thread=%s: %s", table, thread_id, e)


def list_sessions(user_sub: str, tier: str, limit: int = 50) -> list[ChatSession]:
    """List a user's chat sessions, newest first.

    The `limit` caps the number returned (the UI paginates). The tier limit
    is enforced on write (upsert), so the stored set is already within quota.
    """
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT session_id, thread_id, title, last_message, message_count, created_at, updated_at
            FROM agent.chat_sessions
            WHERE user_sub = %s AND archived_at IS NULL
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_sub, limit),
        ).fetchall()

    return [
        ChatSession(
            session_id=r[0],
            thread_id=r[1],
            title=r[2],
            last_message=r[3],
            message_count=r[4],
            created_at=r[5],
            updated_at=r[6],
        )
        for r in rows
    ]


async def get_session_messages(user_sub: str, session_id: str, graph) -> list[dict]:
    """Reconstruct the message history for a session from the checkpointer.

    Returns a list of {role, content, timestamp} dicts. The graph's
    checkpointer stores the `history` channel in its state; we read the
    latest checkpoint and extract the history list.
    """
    thread_id = _thread_id(user_sub, session_id)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph.aget_state(config)
        if state and state.values:
            history = state.values.get("history", [])
            return [
                {
                    "role": h.get("role", "user"),
                    "content": h.get("content", ""),
                    "timestamp": h.get("timestamp", 0),
                }
                for h in history
            ]
    except Exception as e:
        log.warning("[sessions] Failed to load history for thread=%s: %s", thread_id, e)
    return []


def delete_session(user_sub: str, session_id: str) -> bool:
    """Delete a single chat session and its checkpointer state.

    Returns True if a row was deleted, False if the session didn't exist
    (or didn't belong to this user — the WHERE clause enforces ownership).
    """
    pool = get_pool()
    thread_id = _thread_id(user_sub, session_id)
    with pool.connection() as conn:
        row = conn.execute(
            """
            DELETE FROM agent.chat_sessions
            WHERE user_sub = %s AND session_id = %s
            RETURNING id
            """,
            (user_sub, session_id),
        ).fetchone()
        if row is None:
            return False
        _delete_checkpointer_state(conn, thread_id)
        return True


def delete_all_sessions(user_sub: str) -> int:
    """Delete all chat sessions for a user. Returns the number deleted."""
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "DELETE FROM agent.chat_sessions WHERE user_sub = %s RETURNING thread_id",
            (user_sub,),
        ).fetchall()
        for r in rows:
            _delete_checkpointer_state(conn, r[0])
        return len(rows)


def archive_user_sessions(conn, user_sub: str) -> int:
    """Soft-archive all chat sessions for a user and delete their checkpointer state.

    Sets ``archived_at = now()`` on every chat_sessions row for the given
    user_sub (so the sessions no longer appear in the active list) and
    deletes the LangGraph checkpointer state for those threads (freeing
    the message history storage). Returns the number of sessions archived.

    The caller owns the connection (``conn``) so this can run inside a
    larger transaction — e.g. an account-deletion flow that also touches
    other tables.
    """
    rows = conn.execute(
        """
        UPDATE agent.chat_sessions
        SET archived_at = now()
        WHERE user_sub = %s AND archived_at IS NULL
        RETURNING thread_id
        """,
        (user_sub,),
    ).fetchall()
    for r in rows:
        _delete_checkpointer_state(conn, r[0])
    log.info("[sessions] Archived %d session(s) for user=%s", len(rows), user_sub)
    return len(rows)


def list_archived_sessions(limit: int = 200) -> list[ChatSession]:
    """List archived chat sessions across all users (admin only), newest archived first."""
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT session_id, thread_id, title, last_message, message_count, created_at, updated_at
            FROM agent.chat_sessions
            WHERE archived_at IS NOT NULL
            ORDER BY archived_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    return [
        ChatSession(
            session_id=r[0],
            thread_id=r[1],
            title=r[2],
            last_message=r[3],
            message_count=r[4],
            created_at=r[5],
            updated_at=r[6],
        )
        for r in rows
    ]


def get_session_limit(tier: str) -> Optional[int]:
    """Return the max saved sessions for a tier (None = unlimited)."""
    return _tier_limit(tier)
