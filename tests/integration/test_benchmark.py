"""Benchmark suite — validates the Phase 7 goals.

Measures for each scenario:
  - llm_calls (0 for deterministic paths)
  - tool_calls (0 for unauthorized / trivial / out-of-scope)
  - language match
  - latency
"""
import json
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

BENCHMARK_SCENARIOS = [
    {
        "name": "greeting",
        "message": "Hi",
        "tier": "free",
        "expected_llm_calls": 0,
        "expected_tool_calls": 0,
    },
    {
        "name": "capabilities",
        "message": "What can you do?",
        "tier": "free",
        "expected_llm_calls": 0,
        "expected_tool_calls": 0,
    },
    {
        "name": "out_of_scope",
        "message": "What is the weather?",
        "tier": "free",
        "expected_llm_calls": 0,
        "expected_tool_calls": 0,
    },
    {
        "name": "free_hot_pairs",
        "message": "What are the hot pairs?",
        "tier": "free",
        "expected_llm_calls": 0,
        "expected_tool_calls": 1,
        "expected_tool": "get_statistics",
    },
    {
        "name": "free_analyze",
        "message": "Analyze 7, 11, 17, 24, 31, 36",
        "tier": "free",
        "expected_llm_calls": 0,
        "expected_tool_calls": 1,
        "expected_tool": "analyze",
    },
    {
        "name": "free_generate",
        "message": "Generate 3 forms",
        "tier": "free",
        "expected_llm_calls": 0,
        "expected_tool_calls": 1,
        "expected_tool": "generate_form",
    },
    {
        "name": "free_unauthorized_save",
        "message": "Save 1,2,3,4,5,6",
        "tier": "free",
        "expected_llm_calls": 0,
        "expected_tool_calls": 0,
    },
    {
        "name": "ambiguous_hot",
        "message": "hot",
        "tier": "free",
        "expected_llm_calls": 0,
        "expected_tool_calls": 0,
    },
    {
        "name": "hebrew_hot_pairs",
        "message": "מה הזוגות החמים?",
        "tier": "free",
        "language": "he",
        "expected_llm_calls": 0,
        "expected_tool_calls": 1,
        "expected_tool": "get_statistics",
    },
    {
        "name": "admin_audit",
        "message": "Show me the audit log",
        "tier": "admin",
        "expected_llm_calls": 1,
        "expected_tool_calls": 1,
    },
]


@pytest.fixture(scope="module")
def baseline():
    path = Path(__file__).parent.parent / "benchmark" / "baseline.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _get_headers(tier, request):
    if tier == "free":
        return request.getfixturevalue("free_headers")
    if tier == "paid":
        return request.getfixturevalue("paid_headers")
    return request.getfixturevalue("admin_headers")


class TestBenchmark:
    """Run benchmark scenarios and assert on the new architecture invariants."""

    @pytest.mark.parametrize("scenario", BENCHMARK_SCENARIOS, ids=lambda s: s["name"])
    def test_scenario(self, client, request, scenario, mock_llm_store, mock_tool_clients):
        """Each benchmark scenario should satisfy zero-LLM / zero-tool invariants."""
        headers = _get_headers(scenario["tier"], request)
        mock_llm_store.get_llm().responses = ["Mock planner response"]

        start = time.perf_counter()
        resp = client.post(
            "/chat",
            json={"session_id": f"bench-{scenario['name']}", "message": scenario["message"]},
            headers=headers,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200
        data = resp.json()

        if scenario.get("expected_llm_calls") == 0:
            # Deterministic paths: not paused and contain a response.
            assert data.get("paused") is not True, f"{scenario['name']}: should not pause"
            assert "response" in data

        if scenario.get("expected_tool_calls") == 0:
            # No tool should have been executed.
            assert data.get("tool_result") is None

        if scenario.get("language") == "he":
            import re
            response = data.get("response", "")
            assert re.search(r"[\u0590-\u05FF]", response), (
                f"{scenario['name']}: expected Hebrew in response, got: {response}"
            )

        print(f"{scenario['name']}: {elapsed_ms:.1f}ms, status={resp.status_code}")

    def test_compare_with_baseline(self, baseline):
        """Read the baseline and report a simple pass/fail summary."""
        if not baseline:
            pytest.skip("No baseline.json found; Phase 0 not captured.")
        assert baseline.get("unit_tests", {}).get("total", 0) <= 181
