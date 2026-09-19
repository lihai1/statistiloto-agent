"""Shared fixtures for real LLM integration tests.

These tests run against the real Ollama container and pgvector DB on the
Docker network. They do NOT use FakeListChatModel and do NOT wipe the DB.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

# Real services on the docker network.
os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "false"
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_BASE_URL"] = "http://ollama:11434"
os.environ["OLLAMA_MODEL"] = "qwen2.5:0.5b"
os.environ["LOTTERY_GRPC_HOST"] = ""
os.environ["BFF_BASE_URL"] = ""

# DB credentials can be injected from the orchestrator's .env via DB_URI.
DB_URI = os.environ.get(
    "DB_URI",
    "postgresql://statistiloto:change-me-in-prod@db:5432/statistiloto",
)
os.environ["DB_URI"] = DB_URI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture(scope="session")
def db_pool():
    """Create a real psycopg connection pool to the running DB."""
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(DB_URI, min_size=1, max_size=5, open=True)
    yield pool
    pool.close()


@pytest.fixture(autouse=True, scope="function")
def inject_pool(db_pool):
    """Inject the real DB pool into the app."""
    from app.rag.store import set_pool
    from app.config.settings import reload_settings

    reload_settings()
    set_pool(db_pool, close_old=False)
    yield


@pytest.fixture(autouse=True, scope="function")
def real_llm_store(db_pool):
    """Use the real Ollama LLM, not the FakeListChatModel."""
    from app.llm.config_store import LLMConfigStore, set_llm_store

    store = LLMConfigStore(poll_seconds=999, pg_pool=db_pool)
    set_llm_store(store)
    yield store
    store.stop_poller()
    set_llm_store(None)


@pytest.fixture(autouse=True, scope="function")
def real_checkpointer(db_pool):
    """Use durable checkpointer with the real DB for HITL.

    The app endpoints use async graph APIs requiring AsyncPostgresSaver —
    created lazily by get_checkpointer() on the TestClient portal loop.
    Inject None so the app builds its own; clean checkpoint tables so
    HITL state doesn't leak across tests.
    """
    from app.checkpointer import set_checkpointer
    from app.main import reset_graph

    with db_pool.connection() as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name LIKE 'checkpoint%'"
        ).fetchall()
        for (table_name,) in rows:
            conn.execute(f'DELETE FROM "{table_name}"')
    set_checkpointer(None)
    reset_graph()
    yield
    set_checkpointer(None)
    reset_graph()


@pytest.fixture(autouse=True, scope="function")
def real_embeddings():
    """Use the real Ollama embeddings (nomic-embed-text)."""
    from langchain_ollama import OllamaEmbeddings
    from app.rag.retriever import set_embeddings_model
    from app.config.settings import get_settings

    s = get_settings()
    emb = OllamaEmbeddings(
        model=s.rag.embedding_model,
        base_url=s.llm.ollama.base_url,
    )
    set_embeddings_model(emb)
    yield
    from app.rag.retriever import reset_embeddings_model
    reset_embeddings_model()


@pytest.fixture(autouse=True, scope="function")
def mock_tool_clients():
    """Mock external gRPC/HTTP tool clients so the real LLM tests isolate
    the agent discussion flow from the Go/Java services."""
    from app.tools import lottery_grpc, saved_numbers

    lottery_grpc.invalidate_tool_cache()
    lottery_grpc.set_mock_client({
        "generate_form": lambda **kw: {"forms": [[1, 2, 3, 4, 5, 6]]},
        "get_statistics": lambda **kw: {
            "pairs": [
                {"numbers": [1, 2], "count": 10},
                {"numbers": [3, 4], "count": 8},
                {"numbers": [5, 6], "count": 5},
                {"numbers": [1, 3], "count": 4},
                {"numbers": [2, 4], "count": 3},
            ]
        },
        "analyze": lambda **kw: {"matches": [], "frequency": {}},
    })
    saved_numbers.set_mock_client({
        "list_saved_numbers": lambda **kw: {"numbers": []},
        "save_numbers": lambda **kw: {"status": "saved", "id": "mock-001"},
    })
    yield
    lottery_grpc.reset_mock_client()
    saved_numbers.reset_mock_client()


@pytest.fixture
def free_token():
    from app.security import create_test_jwt

    return create_test_jwt(sub="real-llm-free-001", tier="free")


@pytest.fixture
def paid_token():
    from app.security import create_test_jwt

    return create_test_jwt(sub="real-llm-paid-001", tier="paid")


@pytest.fixture
def admin_token():
    from app.security import create_test_jwt

    return create_test_jwt(sub="real-llm-admin-001", tier="admin")


@pytest.fixture
def free_headers(free_token):
    return {"Authorization": f"Bearer {free_token}"}


@pytest.fixture
def paid_headers(paid_token):
    return {"Authorization": f"Bearer {paid_token}"}


@pytest.fixture
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}


@pytest.fixture
def client(real_checkpointer):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
