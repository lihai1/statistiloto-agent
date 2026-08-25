"""Docs ingestion — load markdown product docs into the 'docs' RAG corpus.

Reads .md files from app/rag/docs_source/, splits them into chunks,
embeds each chunk via the configured embeddings model, and inserts
into agent.embeddings.

Content-hash dedup: if a document's content hasn't changed (same hash),
it is skipped. This makes re-indexing idempotent and fast.

Usage:
    python -m app.rag.ingest           # ingest all docs
    python -m app.rag.ingest --force   # clear and re-index all docs
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "docs_source"
CHUNK_SIZE = 500  # characters per chunk (approximate)
CHUNK_OVERLAP = 50  # overlap between chunks for context continuity


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring to break at paragraph/line boundaries."""
    # Split into paragraphs first
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # If the paragraph itself is longer than chunk size, split by sentences/lines
            if len(para) > size:
                lines = para.split("\n")
                current = ""
                for line in lines:
                    if len(current) + len(line) + 1 <= size:
                        current = (current + "\n" + line).strip() if current else line
                    else:
                        if current:
                            chunks.append(current)
                        current = line
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def _content_hash(text: str) -> str:
    """Stable hash of chunk content for dedup."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _get_existing_hashes(corpus: str = "docs") -> set[str]:
    """Get the set of content hashes already indexed in the docs corpus."""
    from app.rag.store import get_pool
    pool = get_pool()
    import json
    hashes = set()
    try:
        with pool.connection() as conn:
            rows = conn.execute(
                "SELECT metadata FROM agent.embeddings WHERE corpus = %s",
                (corpus,),
            ).fetchall()
            for row in rows:
                meta = json.loads(row[0]) if row[0] else {}
                h = meta.get("content_hash")
                if h:
                    hashes.add(h)
    except Exception as e:
        log.warning("Could not read existing hashes: %s", e)
    return hashes


def ingest_docs(force: bool = False, docs_dir: Path | None = None) -> dict:
    """Ingest all markdown docs from docs_source/ into the 'docs' corpus.

    Args:
        force: if True, clear the docs corpus first and re-index everything.
        docs_dir: override the docs directory (for testing).

    Returns:
        {"indexed": N, "skipped": N, "total_chunks": N, "files": [...]}
    """
    from app.rag.retriever import _get_embeddings_model as get_embeddings_model
    from app.rag.indexers import index_document, clear_corpus

    source_dir = docs_dir or DOCS_DIR
    if not source_dir.exists():
        log.warning("Docs source directory not found: %s", source_dir)
        return {"indexed": 0, "skipped": 0, "total_chunks": 0, "files": []}

    if force:
        log.info("Clearing docs corpus for full re-index")
        clear_corpus("docs")

    existing = _get_existing_hashes() if not force else set()
    embeddings = get_embeddings_model()

    md_files = sorted(source_dir.glob("*.md"))
    indexed = 0
    skipped = 0
    total_chunks = 0
    file_names = []

    for md_file in md_files:
        file_names.append(md_file.name)
        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        total_chunks += len(chunks)

        # Embed all chunks for this file at once
        try:
            vectors = embeddings.embed_documents(chunks)
        except Exception as e:
            log.error("Failed to embed chunks for %s: %s", md_file.name, e)
            continue

        for chunk, vector in zip(chunks, vectors):
            chash = _content_hash(chunk)
            if chash in existing:
                skipped += 1
                continue
            metadata = {
                "source": md_file.name,
                "content_hash": chash,
                "chunk_index": total_chunks - len(chunks) + chunks.index(chunk),
            }
            try:
                index_document("docs", chunk, metadata, vector)
                indexed += 1
                existing.add(chash)
            except Exception as e:
                log.error("Failed to index chunk from %s: %s", md_file.name, e)

    log.info("Ingestion complete: indexed=%d skipped=%d total_chunks=%d files=%d",
             indexed, skipped, total_chunks, len(md_files))
    return {
        "indexed": indexed,
        "skipped": skipped,
        "total_chunks": total_chunks,
        "files": file_names,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest product docs into RAG")
    parser.add_argument("--force", action="store_true", help="Clear and re-index all docs")
    args = parser.parse_args()
    result = ingest_docs(force=args.force)
    print(f"Done: {result}")
