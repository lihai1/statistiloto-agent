"""Unit tests for the TTL tool result cache in lottery_grpc.

Verifies that repeated identical tool calls hit the cache (no second gRPC call),
different args miss, TTL expiry re-fetches, invalidation works, and max-size
eviction drops the oldest entry. Uses mock gRPC clients with call counters.
"""

import time

import pytest

from app.tools import lottery_grpc
from app.tools.lottery_grpc import (
    get_statistics,
    generate_form,
    analyze,
    invalidate_tool_cache,
    reset_mock_client,
)


@pytest.fixture(autouse=True)
def _reset():
    """Reset mock + cache before and after each test."""
    reset_mock_client()
    invalidate_tool_cache()
    yield
    reset_mock_client()
    invalidate_tool_cache()


def _make_counting_mock(tool: str, result: dict):
    """Create a mock that records how many times it was called."""
    calls = {"count": 0}

    def _fn(**kw):
        calls["count"] += 1
        return result

    return {tool: _fn}, calls


# ── get_statistics ────────────────────────────────────────────


class TestCacheGetStatistics:
    def test_cache_miss_first_call(self):
        mock, calls = _make_counting_mock("get_statistics", {"groups": []})
        lottery_grpc.set_mock_client(mock)
        get_statistics(group_size=2, strength="hot")
        assert calls["count"] == 1

    def test_cache_hit_second_call(self):
        mock, calls = _make_counting_mock("get_statistics", {"groups": [{"numbers": [1, 2], "count": 5}]})
        lottery_grpc.set_mock_client(mock)
        r1 = get_statistics(group_size=2, strength="hot")
        r2 = get_statistics(group_size=2, strength="hot")
        assert calls["count"] == 1
        assert r1 == r2

    def test_cache_miss_different_args(self):
        mock, calls = _make_counting_mock("get_statistics", {"groups": []})
        lottery_grpc.set_mock_client(mock)
        get_statistics(group_size=2, strength="hot")
        get_statistics(group_size=3, strength="hot")
        assert calls["count"] == 2

    def test_cache_invalidation(self):
        mock, calls = _make_counting_mock("get_statistics", {"groups": []})
        lottery_grpc.set_mock_client(mock)
        get_statistics(group_size=2, strength="hot")
        invalidate_tool_cache()
        get_statistics(group_size=2, strength="hot")
        assert calls["count"] == 2

    def test_cache_ttl_expiry(self, monkeypatch):
        # Set TTL to 0.05s so it expires almost immediately.
        monkeypatch.setattr(lottery_grpc, "_get_cache_ttl", lambda: 0.05)
        mock, calls = _make_counting_mock("get_statistics", {"groups": []})
        lottery_grpc.set_mock_client(mock)
        get_statistics(group_size=2, strength="hot")
        time.sleep(0.1)
        get_statistics(group_size=2, strength="hot")
        assert calls["count"] == 2

    def test_cache_max_size_eviction(self, monkeypatch):
        monkeypatch.setattr(lottery_grpc, "_get_cache_max_size", lambda: 2)
        mock, calls = _make_counting_mock("get_statistics", {"groups": []})
        lottery_grpc.set_mock_client(mock)
        # Insert 3 distinct keys with max_size=2 — the first key is evicted.
        get_statistics(group_size=1, strength="hot")
        get_statistics(group_size=2, strength="hot")
        get_statistics(group_size=3, strength="hot")
        assert calls["count"] == 3
        # gs=1 was evicted when gs=3 was inserted → miss (count=4).
        get_statistics(group_size=1, strength="hot")
        assert calls["count"] == 4
        # gs=3 was most recently added and should still be cached → hit.
        get_statistics(group_size=3, strength="hot")
        assert calls["count"] == 4


# ── generate_form ─────────────────────────────────────────────


class TestCacheGenerateForm:
    def test_cache_hit_second_call(self):
        mock, calls = _make_counting_mock("generate_form", {"forms": [{"numbers": [1, 2, 3]}]})
        lottery_grpc.set_mock_client(mock)
        generate_form(how_many=5, form_type=6, strength=2)
        generate_form(how_many=5, form_type=6, strength=2)
        assert calls["count"] == 1

    def test_cache_miss_different_args(self):
        mock, calls = _make_counting_mock("generate_form", {"forms": []})
        lottery_grpc.set_mock_client(mock)
        generate_form(how_many=5, form_type=6, strength=2)
        generate_form(how_many=10, form_type=6, strength=2)
        assert calls["count"] == 2


# ── analyze ───────────────────────────────────────────────────


class TestCacheAnalyze:
    def test_cache_hit_second_call(self):
        mock, calls = _make_counting_mock("analyze", {"frequency_groups": [], "archive_size": 0})
        lottery_grpc.set_mock_client(mock)
        analyze(form=[1, 2, 3, 4, 5, 6])
        analyze(form=[1, 2, 3, 4, 5, 6])
        assert calls["count"] == 1

    def test_cache_miss_different_args(self):
        mock, calls = _make_counting_mock("analyze", {"frequency_groups": [], "archive_size": 0})
        lottery_grpc.set_mock_client(mock)
        analyze(form=[1, 2, 3, 4, 5, 6])
        analyze(form=[7, 8, 9, 10, 11, 12])
        assert calls["count"] == 2
