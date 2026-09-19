"""Ensure trusted local Ollama models exist on startup.

The ``llm.ollama.models`` list in agent.yaml names the models admins trust
and may select via PUT /llm-config. On agent startup we compare that list
against the local Ollama server's /api/tags and pull anything missing, so a
fresh environment self-heals without manual ``ollama pull`` steps.

Pulls run sequentially in the background — they can be GBs — and failures
are logged, never raised, so they can't block or crash startup.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _norm(name: str) -> str:
    """Normalize a model ref so ``name`` == ``name:latest`` like Ollama does."""
    name = name.strip()
    tail = name.rsplit("/", 1)[-1]
    return name if ":" in tail else f"{name}:latest"


async def ensure_ollama_models(base_url: str, models: list[str]) -> dict:
    """Pull trusted models missing from the local Ollama server.

    Returns ``{"pulled": [...], "present": [...], "failed": [...]}``.
    Fault-tolerant: connectivity/pull errors are logged and reported in
    ``failed`` — the caller never sees an exception.
    """
    import httpx

    wanted = {_norm(m) for m in models if m and m.strip()}
    result = {"pulled": [], "present": [], "failed": []}
    if not wanted:
        return result

    base = base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base}/api/tags")
            resp.raise_for_status()
            local = {_norm(m.get("name", "")) for m in resp.json().get("models", [])}
    except Exception as e:
        log.warning("[startup] Ollama /api/tags unreachable — skipping model check: %s", e)
        result["failed"] = sorted(wanted)
        return result

    # Pulls can take many minutes (multi-GB); run sequentially without a
    # request timeout so a slow download isn't killed mid-flight.
    async with httpx.AsyncClient(timeout=None) as client:
        for name in sorted(wanted):
            if name in local:
                result["present"].append(name)
                continue
            log.info("[startup] pulling trusted Ollama model %s", name)
            try:
                resp = await client.post(f"{base}/api/pull", json={"name": name, "stream": False})
                resp.raise_for_status()
                result["pulled"].append(name)
                log.info("[startup] pulled Ollama model %s", name)
            except Exception as e:
                result["failed"].append(name)
                log.warning("[startup] pull of Ollama model %s failed: %s", name, e)
    log.info("[startup] Ollama model check: present=%d pulled=%d failed=%s",
             len(result["present"]), len(result["pulled"]), result["failed"] or "-")
    return result
