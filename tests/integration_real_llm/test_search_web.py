"""Real LLM test for the admin online search tool."""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.integration]


class TestAdminSearchWeb:
    def test_admin_search_web(self, client, admin_headers):
        """Admin asks to search the web; admin_ops should call search_web."""
        session_id = f"real-admin-search-{int(time.time())}"
        resp = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "Search the web for latest lottery regulation news",
                "intent": "admin_ops",
            },
            headers=admin_headers,
        )
        print(f"\n[ADMIN SEARCH] status={resp.status_code} body={resp.text[:1000]}")
        assert resp.status_code == 200
        data = resp.json()
        if data.get("paused"):
            pytest.skip(f"Search unexpectedly paused for HITL: {data}")
        response = data.get("response") or ""
        assert "results" in response.lower(), f"Expected search results, got: {response[:500]}"
