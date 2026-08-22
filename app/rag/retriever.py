"""Role-scoped, per-tenant RAG retrieval.

CRITICAL: user_data is ALWAYS filtered by user_sub from the JWT —
never trusted to the LLM to self-scope.

EXCEPTION: admin (owner/developer super-user) can see ALL users' user_data
for support/debugging purposes — the user_sub filter is bypassed for admin.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config.settings import get_settings
from app.rag.store import get_pool

log = logging.getLogger(__name__)

# Injectable embeddings model — tests can set this to a mock.
_embeddings_model = None


def set_embeddings_model(model):
    """Inject an embeddings model (for testing)."""
    global _embeddings_model
    _embeddings_model = model


def reset_embeddings_model():
    """Reset to default (Ollama) embeddings."""
    global _embeddings_model
    _embeddings_model = None


def _get_embeddings_model():
    """Get the embedding model (Ollama nomic-embed-text by default)."""
    if _embeddings_model is not None:
        return _embeddings_model
    from langchain_ollama import OllamaEmbeddings
    s = get_settings()
    return OllamaEmbeddings(
        model=s.rag.embedding_model,
        base_url=s.llm.ollama.base_url,
    )


def retrieve(
    query: str,
    corpora: list[str],
    user_sub: str,
    tier: str = "free",
    top_k: int | None = None,
    embedding: list[float] | None = None,
) -> list[dict]:
    """Retrieve relevant chunks from the specified corpora.

    Args:
        query: the search query text.
        corpora: list of corpus names to search (role-scoped by caller).
        user_sub: the JWT user_sub — used to filter user_data.
        tier: the user's tier — admin bypasses user_sub filter on user_data.
        top_k: number of results (default from config).
        embedding: pre-computed embedding vector (for testing without Ollama).

    Returns:
        List of {text, metadata, distance} dicts.
    """
    s = get_settings()
    if top_k is None:
        top_k = s.rag.top_k

    if embedding is None:
        emb_model = _get_embeddings_model()
        embedding = emb_model.embed_query(query)

    pool = get_pool()

    # CRITICAL: user_data is ALWAYS filtered by user_sub from the JWT.
    # EXCEPTION: admin (owner/developer) can see ALL users' user_data.
    if "user_data" in corpora:
        if tier == "admin":
            # Admin sees ALL users' data — no user_sub filter.
            with pool.connection() as conn:
                rows = conn.execute(
                    """SELECT content, metadata, embedding <=> %s::vector AS dist
                       FROM agent.embeddings
                       WHERE corpus = 'user_data'
                       ORDER BY dist LIMIT %s""",
                    (str(embedding), top_k),
                ).fetchall()
        else:
            # Everyone else: always filtered by user_sub from JWT.
            with pool.connection() as conn:
                rows = conn.execute(
                    """SELECT content, metadata, embedding <=> %s::vector AS dist
                       FROM agent.embeddings
                       WHERE corpus = 'user_data' AND metadata->>'user_sub' = %s
                       ORDER BY dist LIMIT %s""",
                    (str(embedding), user_sub, top_k),
                ).fetchall()
    else:
        with pool.connection() as conn:
            rows = conn.execute(
                """SELECT content, metadata, embedding <=> %s::vector AS dist
                   FROM agent.embeddings
                   WHERE corpus = ANY(%s)
                     AND (metadata->>'user_sub' IS NULL OR metadata->>'user_sub' = %s)
                   ORDER BY dist LIMIT %s""",
                (str(embedding), corpora, user_sub, top_k),
            ).fetchall()

    import json
    return [
        {"text": r[0], "metadata": json.loads(r[1]) if isinstance(r[1], str) else r[1], "distance": float(r[2])}
        for r in rows
    ]
