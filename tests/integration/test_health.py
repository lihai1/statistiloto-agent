"""Integration test: health endpoint.

Verifies the FastAPI app starts and /healthz responds.
"""
import pytest

pytestmark = pytest.mark.integration


class TestHealth:
    def test_healthz_returns_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
