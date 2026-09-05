"""Unit tests for LLM config_store.check_connection — no DB / no real LLM needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.llm.config_store import LLMConfig, check_connection


class TestCheckConnection:
    """Tests for check_connection() with mocked httpx responses."""

    @pytest.mark.asyncio
    async def test_mock_provider(self):
        cfg = LLMConfig(provider="mock", model="fake")
        ok, detail = await check_connection(cfg)
        assert ok is True
        assert "Mock" in detail

    @pytest.mark.asyncio
    async def test_ollama_success(self):
        cfg = LLMConfig(provider="ollama", model="llama3", base_url="http://ollama:11434")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama3"}, {"name": "qwen2"}]}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok, detail = await check_connection(cfg)
        assert ok is True
        assert "2 models" in detail

    @pytest.mark.asyncio
    async def test_ollama_failure(self):
        cfg = LLMConfig(provider="ollama", model="llama3", base_url="http://bad:11434")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok, detail = await check_connection(cfg)
        assert ok is False
        assert "Ollama" in detail

    @pytest.mark.asyncio
    async def test_openai_success(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok, detail = await check_connection(cfg)
        assert ok is True
        assert "OpenAI" in detail

    @pytest.mark.asyncio
    async def test_openai_failure(self):
        cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-bad")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=Exception("401 Unauthorized"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok, detail = await check_connection(cfg)
        assert ok is False
        assert "OpenAI" in detail

    @pytest.mark.asyncio
    async def test_anthropic_success(self):
        cfg = LLMConfig(provider="anthropic", model="claude-3", api_key="sk-ant-test")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok, detail = await check_connection(cfg)
        assert ok is True
        assert "Anthropic" in detail

    @pytest.mark.asyncio
    async def test_anthropic_failure(self):
        cfg = LLMConfig(provider="anthropic", model="claude-3", api_key="sk-ant-bad")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=Exception("401"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok, detail = await check_connection(cfg)
        assert ok is False
        assert "Anthropic" in detail

    @pytest.mark.asyncio
    async def test_gemini_success(self):
        cfg = LLMConfig(provider="gemini", model="gemini-2.0-flash", api_key="AIza-test")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok, detail = await check_connection(cfg)
        assert ok is True
        assert "Gemini" in detail

    @pytest.mark.asyncio
    async def test_gemini_failure(self):
        cfg = LLMConfig(provider="gemini", model="gemini-2.0-flash", api_key="AIza-bad")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=Exception("403 Forbidden"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            ok, detail = await check_connection(cfg)
        assert ok is False
        assert "Gemini" in detail

    @pytest.mark.asyncio
    async def test_unknown_provider(self):
        cfg = LLMConfig(provider="unknown", model="x")
        ok, detail = await check_connection(cfg)
        assert ok is False
        assert "Unknown" in detail
