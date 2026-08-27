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
EXAMPLES_DIR = Path(__file__).parent / "examples_source"
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


def _extract_title(text: str) -> str:
    """Extract the first H1 markdown title from text, or empty string."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _make_chunk_header(source_file: str, title: str) -> str:
    """Build a short header prepended to each chunk for better embedding context.

    Format: [source: hot-cold.md] Hot and Cold Numbers
    """
    title_part = f" {title}" if title else ""
    return f"[source: {source_file}]{title_part}\n"


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
        title = _extract_title(text)
        header = _make_chunk_header(md_file.name, title)
        raw_chunks = _chunk_text(text)
        # Prepend the source header to each chunk for better embedding context.
        chunks = [header + c for c in raw_chunks]
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
                "title": title,
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


def _detect_tool_name(assistant_msg: str) -> str | None:
    """Detect a tool name from a 'TOOL: <name> ARGS: ...' line in the assistant message."""
    import re
    if not assistant_msg:
        return None
    m = re.match(r"TOOL:\s*(\w+)", assistant_msg.strip(), re.IGNORECASE)
    return m.group(1) if m else None


def _validate_example_pair(pair: dict, rel_name: str, index: int) -> None:
    """Validate an example pair has required security metadata.

    Raises ValueError (hard error) if:
      - `audience` is missing (fail-closed: not retrievable without audience)
      - `TOOL:` appears in the assistant message but `required_capability` is missing
        (a TOOL example without required_capability could leak to unauthorized tiers)

    Non-TOOL examples without `required_capability` are allowed (treated as null/generic).
    """
    audience = pair.get("audience")
    if not audience:
        raise ValueError(
            f"{rel_name}[{index}]: missing 'audience' metadata — "
            f"every example must declare audience (public or admin). "
            f"Ingestion aborted (fail-closed)."
        )
    if audience not in ("public", "admin"):
        raise ValueError(
            f"{rel_name}[{index}]: invalid audience={audience!r} — "
            f"must be 'public' or 'admin'. Ingestion aborted."
        )

    assistant_msg = (pair.get("assistant") or "").strip()
    tool_name = _detect_tool_name(assistant_msg)
    if tool_name and not pair.get("required_capability"):
        raise ValueError(
            f"{rel_name}[{index}]: assistant contains 'TOOL: {tool_name}' but "
            f"'required_capability' is missing — TOOL examples must declare "
            f"required_capability so RAG can filter by tier. Ingestion aborted (fail-closed)."
        )


def _format_example_chunk(pair: dict) -> str | None:
    """Format a single Q&A example pair into retrievable text.

    Supports optional fields used by the language-separated corpus:
      lang      — language code (en/he) — included as a metadata label
      context   — structured UI context dict — rendered as key=value lines
      approval  — "approved" / "rejected" — marks HITL flow examples
      final     — expected post-approval response (used with `approval`)

    Returns None if the pair is missing required user/assistant fields.
    """
    user_msg = (pair.get("user") or "").strip()
    assistant_msg = (pair.get("assistant") or "").strip()
    if not user_msg or not assistant_msg:
        return None

    parts: list[str] = []

    lang = (pair.get("lang") or "").strip()
    if lang:
        parts.append(f"Language: {lang}")

    ctx = pair.get("context")
    if isinstance(ctx, dict) and ctx:
        ctx_parts = [f"{k}={v}" for k, v in ctx.items()]
        parts.append("Context: " + ", ".join(ctx_parts))

    approval = (pair.get("approval") or "").strip()
    if approval:
        parts.append(f"Approval: {approval}")

    parts.append(f"User: {user_msg}")
    parts.append(f"Assistant: {assistant_msg}")

    final_msg = (pair.get("final") or "").strip()
    if final_msg:
        parts.append(f"Final: {final_msg}")

    return "\n".join(parts)


def _discover_example_files(source_dir: Path) -> list[Path]:
    """Discover YAML example files under source_dir.

    Supports two layouts:
      1. Flat:    source_dir/*.yaml                  (legacy)
      2. Lang:    source_dir/<lang>/*.yaml           (current)

    The language-separated layout is preferred. Files are returned sorted
    by relative path for deterministic ingestion order.
    """
    # Recursive glob for any .yaml file under source_dir (covers both layouts).
    return sorted(source_dir.rglob("*.yaml"))


def ingest_examples(force: bool = False, examples_dir: Path | None = None) -> dict:
    """Ingest Q&A example pairs from YAML files into the 'examples' corpus.

    Each YAML file contains a list of example records. Each record becomes
    a single chunk formatted as:

        [Language: <lang>]
        [Context: key=value, ...]
        [Approval: approved|rejected]
        User: <question>
        Assistant: <expected response or TOOL: line>
        [Final: <post-approval response>]

    The corpus is organized under language subdirectories (en/, he/) so
    retrieval can be filtered by user language. Legacy flat-layout files
    at the source root are still supported.

    This corpus gives the LLM few-shot context at retrieval time — when a
    user asks a similar question, the retrieved example shows the expected
    response style, tool selection, argument format, and approval-flow
    behavior. This reduces LLM reasoning load and improves determinism.

    Args:
        force: if True, clear the examples corpus first and re-index everything.
        examples_dir: override the examples directory (for testing).

    Returns:
        {"indexed": N, "skipped": N, "total_pairs": N, "files": [...]}
    """
    import yaml
    from app.rag.retriever import _get_embeddings_model as get_embeddings_model
    from app.rag.indexers import index_document, clear_corpus

    source_dir = examples_dir or EXAMPLES_DIR
    if not source_dir.exists():
        log.warning("Examples source directory not found: %s", source_dir)
        return {"indexed": 0, "skipped": 0, "total_pairs": 0, "files": []}

    if force:
        log.info("Clearing examples corpus for full re-index")
        clear_corpus("examples")

    existing = _get_existing_hashes("examples") if not force else set()
    embeddings = get_embeddings_model()

    yaml_files = _discover_example_files(source_dir)
    indexed = 0
    skipped = 0
    total_pairs = 0
    file_names = []

    for yml_file in yaml_files:
        rel_name = str(yml_file.relative_to(source_dir))
        file_names.append(rel_name)
        with open(yml_file, encoding="utf-8") as f:
            pairs = yaml.safe_load(f)
        if not isinstance(pairs, list):
            log.warning("Skipping %s: expected a YAML list", rel_name)
            continue

        # Validate each pair has required security metadata (fail-closed).
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                continue
            _validate_example_pair(pair, rel_name, i)

        # Build chunk text for each example pair, preserving language,
        # context, and approval-flow metadata.
        chunks: list[str] = []
        chunk_langs: list[str] = []
        chunk_audiences: list[str] = []
        chunk_capabilities: list[str | None] = []
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            chunk = _format_example_chunk(pair)
            if chunk is None:
                continue
            chunks.append(chunk)
            chunk_langs.append((pair.get("lang") or "").strip())
            chunk_audiences.append((pair.get("audience") or "").strip())
            chunk_capabilities.append(pair.get("required_capability"))

        total_pairs += len(chunks)
        if not chunks:
            continue

        try:
            vectors = embeddings.embed_documents(chunks)
        except Exception as e:
            log.error("Failed to embed examples for %s: %s", rel_name, e)
            continue

        for chunk, vector, lang, audience, capability in zip(
            chunks, vectors, chunk_langs, chunk_audiences, chunk_capabilities
        ):
            chash = _content_hash(chunk)
            if chash in existing:
                skipped += 1
                continue
            metadata = {
                "source": rel_name,
                "content_hash": chash,
                "type": "qa_example",
            }
            if lang:
                metadata["lang"] = lang
            if audience:
                metadata["audience"] = audience
            # required_capability: null for generic examples, tool name for TOOL examples.
            # Stored as a string or explicitly absent (null is not stored by psycopg JSON).
            if capability is not None:
                metadata["required_capability"] = capability
            try:
                index_document("examples", chunk, metadata, vector)
                indexed += 1
                existing.add(chash)
            except Exception as e:
                log.error("Failed to index example from %s: %s", rel_name, e)

    log.info("Examples ingestion complete: indexed=%d skipped=%d total_pairs=%d files=%d",
             indexed, skipped, total_pairs, len(yaml_files))
    return {
        "indexed": indexed,
        "skipped": skipped,
        "total_pairs": total_pairs,
        "files": file_names,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest product docs and examples into RAG")
    parser.add_argument("--force", action="store_true", help="Clear and re-index everything")
    parser.add_argument("--examples-only", action="store_true", help="Only ingest examples, not docs")
    args = parser.parse_args()
    results = {}
    if not args.examples_only:
        results["docs"] = ingest_docs(force=args.force)
    results["examples"] = ingest_examples(force=args.force)
    print(f"Done: {results}")
