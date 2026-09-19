"""Unit tests for app.llm.model_pull.ensure_ollama_models."""

import pytest

from app.llm.model_pull import _norm, ensure_ollama_models


class _Resp:
    def __init__(self, data=None, status=200):
        self._data = data or {}
        self.status_code = status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _client(tags=None, tags_exc=None, fail_pull=()):
    """Fake httpx.AsyncClient recording POST /api/pull calls."""
    calls = []

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            if tags_exc:
                raise tags_exc
            return _Resp({"models": [{"name": n} for n in (tags or [])]})

        async def post(self, url, json=None, **kw):
            name = json["name"]
            calls.append(name)
            if name in fail_pull:
                return _Resp(status=500)
            return _Resp({"status": "success"})

    return calls, FakeClient


@pytest.mark.asyncio
async def test_empty_list_noop(monkeypatch):
    calls, client = _client()
    monkeypatch.setattr("httpx.AsyncClient", client)
    res = await ensure_ollama_models("http://x", [])
    assert res == {"pulled": [], "present": [], "failed": []}
    assert calls == []


@pytest.mark.asyncio
async def test_skips_present_pulls_missing(monkeypatch):
    calls, client = _client(tags=["a:latest", "b:7b"])
    monkeypatch.setattr("httpx.AsyncClient", client)
    res = await ensure_ollama_models("http://x/", ["a", "b:7b", "c:3b"])
    assert calls == ["c:3b"]                       # normalized "a" == "a:latest"
    assert res["pulled"] == ["c:3b"]
    assert sorted(res["present"]) == ["a:latest", "b:7b"]
    assert res["failed"] == []


@pytest.mark.asyncio
async def test_tags_unreachable_marks_failed_no_raise(monkeypatch):
    calls, client = _client(tags_exc=RuntimeError("conn refused"))
    monkeypatch.setattr("httpx.AsyncClient", client)
    res = await ensure_ollama_models("http://x", ["a:1b", "b:2b"])
    assert calls == []
    assert res["failed"] == ["a:1b", "b:2b"]


@pytest.mark.asyncio
async def test_pull_failure_isolated_per_model(monkeypatch):
    calls, client = _client(tags=[], fail_pull=("bad:1b",))
    monkeypatch.setattr("httpx.AsyncClient", client)
    res = await ensure_ollama_models("http://x", ["ok:1b", "bad:1b"])
    assert sorted(calls) == ["bad:1b", "ok:1b"]
    assert res["pulled"] == ["ok:1b"]
    assert res["failed"] == ["bad:1b"]


def test_norm_adds_latest_when_untagged():
    assert _norm("dicta-instruct-1.7b") == "dicta-instruct-1.7b:latest"
    assert _norm("dicta-instruct-1.7b:latest") == "dicta-instruct-1.7b:latest"
    # registry-style names keep their tag comparison on the last path segment
    assert _norm("dicta-il/DictaLM-3.0-1.7B-Thinking") == "dicta-il/DictaLM-3.0-1.7B-Thinking:latest"
    assert _norm("dicta-il/DictaLM-3.0-1.7B-Thinking:q4_k_m") == "dicta-il/DictaLM-3.0-1.7B-Thinking:q4_k_m"
