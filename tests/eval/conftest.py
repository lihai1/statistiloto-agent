"""Shared fixtures for eval tests.

These tests verify NL→tool-arg mapping, grounding, language matching, and security.
They use the same DB + mock infrastructure as integration tests but assert on
tool arguments and response content rather than just HTTP 200.
"""
import os
import sys
from pathlib import Path

# Set test env vars BEFORE importing app modules.
os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "llama3.1:8b"
os.environ["DB_URI"] = "postgresql://postgres:postgres@localhost:5433/statistiloto"
os.environ["LOTTERY_GRPC_HOST"] = ""
os.environ["BFF_BASE_URL"] = ""

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Reuse the integration conftest fixtures.
from tests.integration.conftest import *  # noqa: F401, F403

import pytest


@pytest.fixture
def set_llm_responses(mock_llm_store):
    """Return a callable that sets mock LLM responses and rebuilds the LLM.

    Usage:
        def test_x(set_llm_responses, client, paid_headers):
            set_llm_responses(["TOOL: get_statistics ARGS: {...}", "final response"])
            resp = client.post("/chat", ...)
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    def _set(responses: list[str]):
        mock_llm_store._mock_responses = responses
        mock_llm_store._llm = FakeListChatModel(responses=responses)

    return _set
