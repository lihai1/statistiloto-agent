"""RAG indexers — populate the agent.embeddings table.

Three corpora:
  - docs: static product docs (markdown)
  - lottery_history: draw summaries (refreshed after scraper runs)
  - user_data: per-user saved numbers + analysis history
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from app.rag.store import get_pool

log = logging.getLogger(__name__)


def _hash_id(corpus: str, content: str) -> str:
    return hashlib.sha256(f"{corpus}:{content[:200]}".encode()).hexdigest()[:16]


def index_document(corpus: str, content: str, metadata: dict, embedding: list[float]) -> str:
    """Insert or update a document in the embeddings table. Returns the doc hash id."""
    doc_id = _hash_id(corpus, content)
    pool = get_pool()
    import json
    with pool.connection() as conn:
        conn.execute(
            """INSERT INTO agent.embeddings (corpus, content, metadata, embedding)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT DO NOTHING
               RETURNING id""",
            (corpus, content, json.dumps(metadata), str(embedding)),
        )
    return doc_id


def index_docs_batch(items: list[tuple[str, dict, list[float]]]) -> int:
    """Batch-index documents. Each item is (content, metadata, embedding) for the 'docs' corpus."""
    pool = get_pool()
    import json
    count = 0
    with pool.connection() as conn:
        for content, metadata, embedding in items:
            conn.execute(
                """INSERT INTO agent.embeddings (corpus, content, metadata, embedding)
                   VALUES ('docs', %s, %s, %s)""",
                (content, json.dumps(metadata), str(embedding)),
            )
            count += 1
    return count


def clear_corpus(corpus: str):
    """Delete all embeddings for a corpus."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM agent.embeddings WHERE corpus = %s", (corpus,))


def index_user_data(user_sub: str, content: str, metadata: dict, embedding: list[float]):
    """Index a user-specific document. Always tagged with user_sub in metadata."""
    metadata = {**metadata, "user_sub": user_sub}
    return index_document("user_data", content, metadata, embedding)
