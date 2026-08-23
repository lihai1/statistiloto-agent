"""Runtime LLM config store — DB-backed, admin-reconfigurable, hot-reload.

Single global LLM for all tiers and all agents. The active provider+model
is resolved at runtime:

1. Boot default from agent.yaml + env.
2. Runtime override from agent.llm_config DB table (set by admin).
   If the table has a row, it wins over env/YAML.

config_store polls the DB every N seconds and hot-swaps the LLM instance.

For tests, set LLM_MOCK=true to use FakeListChatModel (no real LLM calls).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.config.settings import get_settings

log = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str    # "ollama" | "gemini" | "mock"
    model: str
    base_url: str = ""
    api_key: str = ""
    request_timeout_seconds: int = 300


class LLMConfigStore:
    """Reads agent.llm_config table for the active global LLM.

    If the table has a row, it overrides env/YAML boot defaults.
    Polls every N seconds and hot-swaps the LLM instance — no agent restart.
    """

    def __init__(
        self,
        poll_seconds: int = 10,
        pg_pool=None,
        mock_responses: list[str] | None = None,
    ):
        self._lock = threading.Lock()
        self._llm = None
        self._cfg: Optional[LLMConfig] = None
        self._poll_seconds = poll_seconds
        self._pg_pool = pg_pool          # injected for testing
        self._mock_responses = mock_responses
        self._poller_thread: Optional[threading.Thread] = None
        self._running = False
        self._refresh()                  # initial load

    def _get_pool(self):
        if self._pg_pool:
            return self._pg_pool
        from app.rag.store import pg_pool
        return pg_pool

    def _boot_default(self) -> LLMConfig:
        """Fallback from env/YAML when no DB row exists."""
        s = get_settings()
        timeout = s.llm.request_timeout_seconds
        if s.llm.mock:
            return LLMConfig(provider="mock", model="fake-list", request_timeout_seconds=timeout)
        provider = s.llm.provider
        if provider == "ollama":
            return LLMConfig(
                provider="ollama",
                model=s.llm.ollama.model,
                base_url=s.llm.ollama.base_url,
                request_timeout_seconds=timeout,
            )
        return LLMConfig(
            provider="gemini",
            model=s.llm.gemini.model,
            api_key=s.llm.gemini.api_key,
            request_timeout_seconds=timeout,
        )

    def _read_db(self) -> Optional[LLMConfig]:
        """Read the latest llm_config row from the DB. Returns None if no row or error."""
        try:
            pool = self._get_pool()
            if pool is None:
                return None
            with pool.connection() as conn:
                # Try with request_timeout_seconds column; fall back without it
                try:
                    row = conn.execute(
                        "SELECT provider, model, base_url, api_key, request_timeout_seconds "
                        "FROM agent.llm_config ORDER BY updated_at DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        return LLMConfig(
                            provider=row[0],
                            model=row[1],
                            base_url=row[2] or "",
                            api_key=row[3] or "",
                            request_timeout_seconds=row[4] or 300,
                        )
                except Exception:
                    # Column doesn't exist yet (pre-migration) — fall back
                    row = conn.execute(
                        "SELECT provider, model, base_url, api_key "
                        "FROM agent.llm_config ORDER BY updated_at DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        return LLMConfig(
                            provider=row[0],
                            model=row[1],
                            base_url=row[2] or "",
                            api_key=row[3] or "",
                        )
        except Exception as e:
            log.debug("Failed to read llm_config from DB: %s", e)
        return None

    def _build_llm(self, cfg: LLMConfig):
        """Build a chat model instance from config."""
        timeout = cfg.request_timeout_seconds or 300

        if cfg.provider == "mock":
            from langchain_core.language_models.fake_chat_models import FakeListChatModel
            responses = self._mock_responses or ["Mock LLM response"]
            return FakeListChatModel(responses=responses)

        if cfg.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=cfg.model,
                base_url=cfg.base_url or "http://ollama:11434",
                timeout=timeout,
            )

        if cfg.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=cfg.model,
                google_api_key=cfg.api_key,
                timeout=timeout,
            )

        raise ValueError(f"Unknown LLM provider: {cfg.provider}")

    def _refresh(self):
        """Reload config from DB (or boot default) and rebuild the LLM."""
        cfg = self._read_db() or self._boot_default()
        try:
            llm = self._build_llm(cfg)
        except Exception as e:
            log.warning("Failed to build LLM (provider=%s model=%s): %s — falling back to mock",
                        cfg.provider, cfg.model, e)
            cfg = LLMConfig(provider="mock", model="fallback", request_timeout_seconds=cfg.request_timeout_seconds)
            llm = self._build_llm(cfg)

        with self._lock:
            self._cfg = cfg
            self._llm = llm
        log.info("LLM config loaded: provider=%s model=%s", cfg.provider, cfg.model)

    def start_poller(self):
        """Start the background DB poller (called at app startup)."""
        if self._running:
            return
        self._running = True

        def loop():
            while self._running:
                time.sleep(self._poll_seconds)
                try:
                    self._refresh()
                except Exception as e:
                    log.warning("LLM config poll failed: %s", e)

        self._poller_thread = threading.Thread(target=loop, daemon=True, name="llm-config-poller")
        self._poller_thread.start()

    def stop_poller(self):
        """Stop the background poller (for tests)."""
        self._running = False
        if self._poller_thread:
            self._poller_thread.join(timeout=5)

    def get_llm(self):
        """Return the current global LLM instance."""
        with self._lock:
            return self._llm

    def get_config(self) -> LLMConfig:
        """Return the current LLM config."""
        with self._lock:
            return self._cfg

    def force_refresh(self):
        """Force an immediate refresh (for testing after DB writes)."""
        self._refresh()


# ── Module-level singleton ───────────────────────────────────
# Initialized lazily — tests can replace it before first use.

_llm_store: Optional[LLMConfigStore] = None


def get_llm_store() -> LLMConfigStore:
    """Get or create the singleton LLMConfigStore."""
    global _llm_store
    if _llm_store is None:
        s = get_settings()
        _llm_store = LLMConfigStore(poll_seconds=s.llm.config_poll_seconds)
    return _llm_store


def set_llm_store(store: LLMConfigStore):
    """Replace the singleton (for testing)."""
    global _llm_store
    if _llm_store:
        _llm_store.stop_poller()
    _llm_store = store


def get_llm():
    """All workers call this — returns the single global LLM (no tier argument)."""
    return get_llm_store().get_llm()
