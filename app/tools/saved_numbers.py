"""HTTP client to the Java BFF for saved numbers CRUD."""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from app.config.settings import get_settings

log = logging.getLogger(__name__)


def _get_base_url() -> str:
    return get_settings().bff.base_url


def list_saved_numbers(user_sub: str, jwt_token: str) -> dict:
    """GET /api/user/numbers — list the user's saved numbers."""
    if _mock_client is not None:
        fn = _mock_client.get("list_saved_numbers")
        if fn:
            return fn(user_sub=user_sub, jwt_token=jwt_token)
        return {"numbers": [], "error": "Mock list_saved_numbers not configured"}

    base = _get_base_url()
    if not base:
        return {"numbers": [], "error": "BFF base URL not configured"}
    with httpx.Client() as client:
        resp = client.get(
            f"{base}/api/user/numbers",
            headers={"Authorization": f"Bearer {jwt_token}"},
        )
        resp.raise_for_status()
        return resp.json()


def save_numbers(user_sub: str, jwt_token: str, category: str,
                 numbers: list[int], will_be: list[int] | None = None) -> dict:
    """POST /api/user/numbers — save a set of numbers."""
    if _mock_client is not None:
        fn = _mock_client.get("save_numbers")
        if fn:
            return fn(user_sub=user_sub, jwt_token=jwt_token, category=category,
                      numbers=numbers, will_be=will_be)
        return {"error": "Mock save_numbers not configured"}

    base = _get_base_url()
    if not base:
        return {"error": "BFF base URL not configured"}
    with httpx.Client() as client:
        resp = client.post(
            f"{base}/api/user/numbers",
            headers={"Authorization": f"Bearer {jwt_token}"},
            json={
                "category": category,
                "numbers": numbers,
                "willBe": will_be or [],
            },
        )
        resp.raise_for_status()
        return resp.json()


# ── Mock support for testing ─────────────────────────────────

_mock_client: Optional[dict] = None


def set_mock_client(mock: dict):
    global _mock_client
    _mock_client = mock


def reset_mock_client():
    global _mock_client
    _mock_client = None
