"""Shared test fixtures for integration tests.

Uses real PostgreSQL (pgvector) from docker-compose-dev.yml.
Mocks the LLM (FakeListChatModel) and external services (Go gRPC, Java BFF).
"""

from __future__ import annotations

import os
import sys
import time
import json
import logging
from pathlib import Path

import psycopg
import pytest

log = logging.getLogger(__name__)

# Set test env vars BEFORE importing app modules.
# NOTE: OLLAMA_MODEL is intentionally NOT set here — the e2e_llm fixture
# auto-discovers it from the provider's /api/tags endpoint. For mock-based
# tests the model value is irrelevant (FakeListChatModel ignores it).
os.environ.setdefault("JWT_VERIFY", "false")
os.environ.setdefault("LLM_MOCK", "true")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("DB_URI", "postgresql://postgres:postgres@localhost:5433/statistiloto")
os.environ.setdefault("LOTTERY_GRPC_HOST", "")  # no Go service in tests
os.environ.setdefault("BFF_BASE_URL", "")       # no Java BFF in tests

# Ensure app is importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config.settings import reload_settings, get_settings
from app.security import create_test_jwt
from app.llm.config_store import LLMConfigStore, set_llm_store
from app.rag.store import set_pool
from app.checkpointer import set_checkpointer, reset_checkpointer
from app.main import app, set_graph, reset_graph


# ── DB fixtures ──────────────────────────────────────────────

DB_URI = "postgresql://postgres:postgres@localhost:5433/statistiloto"


@pytest.fixture(scope="session")
def db_pool():
    """Create a real psycopg connection pool to the test DB."""
    from psycopg_pool import ConnectionPool
    pool = ConnectionPool(DB_URI, min_size=1, max_size=5, open=True)
    yield pool
    pool.close()


@pytest.fixture(scope="session")
def db_conn(db_pool):
    """A direct connection for setup/teardown."""
    with db_pool.connection() as conn:
        yield conn


@pytest.fixture(autouse=True, scope="function")
def clean_db(db_pool):
    """Clean all agent tables before each test."""
    with db_pool.connection() as conn:
        conn.execute("DELETE FROM agent.token_usage")
        conn.execute("DELETE FROM agent.audit_log")
        conn.execute("DELETE FROM agent.llm_config")
        conn.execute("DELETE FROM agent.embeddings")
        conn.execute("DELETE FROM agent.chat_sessions")
    yield


# ── Pool injection ───────────────────────────────────────────

@pytest.fixture(autouse=True, scope="function")
def inject_pool(db_pool):
    """Inject the real test DB pool into the app.

    Uses close_old=False so the session-scoped pool is not closed
    between tests — it's only closed at session teardown.
    """
    set_pool(db_pool, close_old=False)
    yield
    # Don't close — the session-scoped db_pool fixture owns the lifecycle.


# ── LLM mock ─────────────────────────────────────────────────

def _discover_ollama_model(base_url: str) -> str:
    """Query Ollama /api/tags and pick the smallest available chat model.

    Prefers models with 'tools' capability, then falls back to the smallest
    model by size. Skips embedding-only models.
    """
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception:
        return "qwen2.5:0.5b"  # fallback

    models = []
    for m in data.get("models", []):
        name = m.get("name", "") or m.get("model", "")
        if not name:
            continue
        caps = m.get("capabilities") or []
        # Skip embedding-only models.
        if caps and "embedding" in caps and "completion" not in caps and "tools" not in caps:
            continue
        size = m.get("size", 0) or 0
        models.append((name, size, caps))

    if not models:
        return "qwen2.5:0.5b"

    # Prefer models with 'tools' capability (needed for analyst/admin flows).
    tools_models = [m for m in models if "tools" in m[2]]
    if tools_models:
        # Pick the smallest tools-capable model for speed.
        tools_models.sort(key=lambda m: m[1] or float("inf"))
        return tools_models[0][0]

    # Otherwise pick the smallest model.
    models.sort(key=lambda m: m[1] or float("inf"))
    return models[0][0]


@pytest.fixture(autouse=True, scope="function")
def mock_llm_store(request, db_pool):
    """Inject an LLM config store.

    For regular integration tests: uses FakeListChatModel (mock).
    For e2e_llm-marked tests: uses a real Ollama LLM instance, with the
    model auto-discovered from the provider's /api/tags endpoint.
    """
    # e2e_llm tests use a real Ollama LLM instead of the mock.
    if request.node.get_closest_marker("e2e_llm"):
        from app.llm.config_store import build_llm, LLMConfig
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        # Auto-discover the model from Ollama, unless explicitly overridden.
        model = os.environ.get("OLLAMA_MODEL")
        if not model:
            model = _discover_ollama_model(ollama_url)
            log.info("[conftest] Auto-discovered Ollama model: %s", model)
        store = LLMConfigStore(
            poll_seconds=999,
            pg_pool=db_pool,
            mock_responses=None,
        )
        store._cfg = LLMConfig(provider="ollama", model=model, base_url=ollama_url)
        store._llm = build_llm(store._cfg)
        set_llm_store(store)
        yield store
        store.stop_poller()
        set_llm_store(None)
        return

    # Default: mock LLM
    store = LLMConfigStore(
        poll_seconds=999,  # don't poll during tests
        pg_pool=db_pool,
        mock_responses=["Mock analysis response"],
    )
    set_llm_store(store)
    yield store
    store.stop_poller()
    set_llm_store(None)


# ── Checkpointer ─────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="function")
def test_checkpointer(db_pool):
    """Use PostgresSaver with the test DB for durable HITL.

    PostgresSaver.from_conn_string() returns a context manager —
    we enter it to get the actual saver instance.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    cm = PostgresSaver.from_conn_string(DB_URI)
    cp = cm.__enter__()
    cp.setup()
    set_checkpointer(cp)
    yield cp
    set_checkpointer(None)
    try:
        cm.__exit__(None, None, None)
    except Exception:
        pass


# ── Mock embeddings ──────────────────────────────────────────

class MockEmbeddings:
    """Mock embeddings model — returns a fixed 768-dim vector."""
    def embed_query(self, text: str) -> list[float]:
        # Return a deterministic vector based on text hash.
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        vec = [float(b) / 255.0 for b in h]  # 16 values
        return vec + [0.0] * (768 - 16)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]


@pytest.fixture(autouse=True, scope="function")
def mock_embeddings(request):
    """Inject an embeddings model.

    For regular integration tests: uses MockEmbeddings (hash-based, no Ollama).
    For e2e_llm-marked tests: uses real OllamaEmbeddings (nomic-embed-text)
    so retrieval quality is testable.

    Yields the model instance so tests that need to compute embeddings
    for direct DB insertion can use it.
    """
    from app.rag.retriever import set_embeddings_model, reset_embeddings_model

    if request.node.get_closest_marker("e2e_llm"):
        from langchain_ollama import OllamaEmbeddings
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_url)
        log.info("[conftest] Using real OllamaEmbeddings (nomic-embed-text)")
    else:
        model = MockEmbeddings()

    set_embeddings_model(model)
    yield model
    reset_embeddings_model()


@pytest.fixture(autouse=True, scope="function")
def ingest_real_docs(request, mock_embeddings, inject_pool, clean_db):
    """Ingest product docs into the 'docs' corpus before each e2e_llm test.

    Uses the real embeddings model (from mock_embeddings fixture) so chunks
    are embedded with nomic-embed-text and retrieval quality is realistic.
    For non-e2e_llm tests, this is a no-op (mock tests don't need docs).

    Depends on inject_pool and clean_db so the DB pool is ready and clean
    before ingestion.
    """
    if not request.node.get_closest_marker("e2e_llm"):
        yield
        return

    from app.rag.ingest import ingest_docs, ingest_examples
    docs_result = ingest_docs(force=True)
    examples_result = ingest_examples(force=True)
    log.info("[conftest] Ingested docs=%s examples=%s for e2e_llm test", docs_result, examples_result)
    yield


@pytest.fixture(autouse=True, scope="function")
def reset_app_graph():
    """Reset the graph between tests so it picks up the new checkpointer."""
    reset_graph()
    yield
    reset_graph()


# ── Mock tool clients ────────────────────────────────────────

@pytest.fixture(autouse=True, scope="function")
def mock_tool_clients():
    """Inject mock lottery gRPC and saved_numbers clients so tools don't
    try to connect to real Go/Java services during integration tests."""
    from app.tools import lottery_grpc, saved_numbers

    lottery_grpc.set_mock_client({
        "generate_form": lambda **kw: {"forms": [{"numbers": [1, 2, 3, 4, 5, 6], "strong": 7}]},
        "get_statistics": lambda **kw: {"groups": [{"numbers": [1, 2], "count": 10}]},
        "analyze": lambda **kw: {
            "frequency_groups": [
                {"size": 1, "combos": 37, "entries": [{"numbers": [7], "count": 5}]},
                {"size": 2, "combos": 666, "entries": [{"numbers": [7, 17], "count": 3}]},
            ],
            "archive_size": 100,
        },
    })
    saved_numbers.set_mock_client({
        "list_saved_numbers": lambda **kw: {"numbers": []},
        "save_numbers": lambda **kw: {"status": "saved", "id": "mock-001"},
    })
    yield
    lottery_grpc.reset_mock_client()
    saved_numbers.reset_mock_client()


# ── JWT tokens ───────────────────────────────────────────────

@pytest.fixture
def free_token():
    return create_test_jwt(sub="free-user-001", tier="free")


@pytest.fixture
def paid_token():
    return create_test_jwt(sub="paid-user-001", tier="paid")


@pytest.fixture
def admin_token():
    return create_test_jwt(sub="admin-user-001", tier="admin")


@pytest.fixture
def free_headers(free_token):
    return {"Authorization": f"Bearer {free_token}"}


@pytest.fixture
def paid_headers(paid_token):
    return {"Authorization": f"Bearer {paid_token}"}


@pytest.fixture
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


# ── Test client ──────────────────────────────────────────────

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
