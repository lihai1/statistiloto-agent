"""Online web search tool.

Searches the public web. Admins can use it for research or fact-checking.
By default it uses the DuckDuckGo HTML interface (no API key required).
A custom search endpoint can be configured via the SEARCH_BASE_URL and
SEARCH_API_KEY environment variables.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.parse
from typing import Optional

import httpx

log = logging.getLogger(__name__)

_mock_client = None


def _parse_ddg_html(html: str) -> list[dict]:
    """Parse DuckDuckGo HTML result page into title/url/snippet list."""
    results = []
    # DuckDuckGo HTML result links have a specific pattern.
    for m in re.finditer(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        re.IGNORECASE | re.DOTALL,
    ):
        href = m.group(1)
        # DDG links are redirects; extract the real URL from the uddg parameter.
        real_url = urllib.parse.unquote(
            re.search(r"uddg=([^&]+)", href).group(1) if re.search(r"uddg=([^&]+)", href) else href
        )
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        results.append({"title": title, "url": real_url, "snippet": ""})
    return results


def _duckduckgo_search(query: str, limit: int) -> list[dict]:
    """Search DuckDuckGo HTML endpoint and parse results."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}&kl=us-en"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Statistiloto/1.0)"}

    with httpx.Client(timeout=15.0, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return _parse_ddg_html(resp.text)[:limit]


def _custom_search(base_url: str, api_key: str | None, query: str, limit: int) -> list[dict]:
    """Call a custom search endpoint (e.g. Tavily, Google CSE, etc.)."""
    params = {"q": query, "limit": limit}
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=30.0, follow_redirects=True, headers=headers) as client:
        resp = client.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if isinstance(data, list):
        return data[:limit]
    if isinstance(data, dict) and "results" in data:
        return data["results"][:limit]
    return []


def search_web(query: str, limit: int = 5) -> dict:
    """Search the web and return a list of results.

    Args:
        query: the search query string.
        limit: max number of results (default 5).
    """
    if _mock_client is not None:
        return _mock_client(query=query, limit=limit)

    base_url = os.environ.get("SEARCH_BASE_URL")
    api_key = os.environ.get("SEARCH_API_KEY")

    try:
        if base_url:
            results = _custom_search(base_url, api_key, query, limit)
        else:
            results = _duckduckgo_search(query, limit)
        log.info("[search_web] query=%s results=%d", query, len(results))
        return {"results": results}
    except Exception as e:
        log.warning("[search_web] failed: %s", e)
        return {"results": [], "error": str(e)}


def set_mock_client(mock):
    """Inject a mock search function for testing."""
    global _mock_client
    _mock_client = mock


def reset_mock_client():
    global _mock_client
    _mock_client = None
