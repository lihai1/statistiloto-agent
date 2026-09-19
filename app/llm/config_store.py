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
    num_predict: int | None = None  # max tokens to generate (None = model default)
    context_window_size: int | None = None  # context window size in tokens (None = model default)


# ── API key hygiene ─────────────────────────────────────────
# Stored configs may reference an env var instead of embedding a
# plaintext key: api_key="env:GOOGLE_API_KEY". Read endpoints mask
# the stored value, and writes treat the mask as "keep existing".
MASKED_API_KEY = "********"


def mask_api_key(key: str | None) -> str:
    """Display form of a stored API key — never reveals the value."""
    return MASKED_API_KEY if key else ""


def resolve_api_key(key: str | None) -> str:
    """Resolve a stored API key for actual use.

    Supports ``env:VAR_NAME`` references so secrets can live in the
    process environment instead of the DB.
    """
    if not key:
        return ""
    if key.startswith("env:"):
        import os
        return os.environ.get(key[4:], "")
    return key


def build_llm(cfg: LLMConfig, mock_responses: list[str] | None = None):
    """Build a chat model instance from config.

    Module-level function so it can be called without a full LLMConfigStore
    (e.g. for the test-connection endpoint).
    """
    timeout = cfg.request_timeout_seconds or 300

    if cfg.provider == "mock":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        responses = mock_responses or ["Mock LLM response"]
        return FakeListChatModel(responses=responses)

    if cfg.provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs: dict = {
            "model": cfg.model,
            "base_url": cfg.base_url or "http://ollama:11434",
            "timeout": timeout,
        }
        if cfg.num_predict is not None:
            kwargs["num_predict"] = cfg.num_predict
        if cfg.context_window_size is not None:
            kwargs["num_ctx"] = cfg.context_window_size
        return ChatOllama(**kwargs)

    if cfg.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_kwargs: dict = {
            "model": cfg.model,
            "google_api_key": resolve_api_key(cfg.api_key),
            "timeout": timeout,
        }
        if cfg.context_window_size is not None:
            gemini_kwargs["max_output_tokens"] = cfg.context_window_size
        return ChatGoogleGenerativeAI(**gemini_kwargs)

    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI
        openai_kwargs: dict = {
            "model": cfg.model,
            "api_key": resolve_api_key(cfg.api_key),
            "base_url": cfg.base_url or "https://api.openai.com/v1",
            "timeout": timeout,
        }
        if cfg.context_window_size is not None:
            openai_kwargs["max_tokens"] = cfg.context_window_size
        return ChatOpenAI(**openai_kwargs)

    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        anthropic_kwargs: dict = {
            "model": cfg.model,
            "api_key": resolve_api_key(cfg.api_key),
            "base_url": cfg.base_url or "https://api.anthropic.com",
            "timeout": timeout,
        }
        if cfg.context_window_size is not None:
            anthropic_kwargs["max_tokens"] = cfg.context_window_size
        return ChatAnthropic(**anthropic_kwargs)

    raise ValueError(f"Unknown LLM provider: {cfg.provider}")


async def check_connection(cfg: LLMConfig) -> tuple[bool, str]:
    """Test that an LLM provider is reachable without sending an inference request.

    Uses lightweight HTTP GET endpoints (model lists / tags) to verify
    connectivity and credentials. Returns ``(True, detail)`` on success or
    ``(False, error_msg)`` on failure.

    Args:
        cfg: the LLM config to test (provider, base_url, api_key).
    """
    import httpx

    provider = cfg.provider
    api_key = resolve_api_key(cfg.api_key)

    if provider == "mock":
        return True, "Mock provider"

    if provider == "ollama":
        base_url = (cfg.base_url or "http://ollama:11434").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                n_models = len(data.get("models", []))
                return True, f"Ollama connected, {n_models} models available"
        except Exception as e:
            return False, f"Ollama connection failed: {e}"

    if provider == "openai":
        base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{base_url}/models", headers=headers)
                resp.raise_for_status()
                return True, "OpenAI connected"
        except Exception as e:
            return False, f"OpenAI connection failed: {e}"

    if provider == "anthropic":
        try:
            headers = {}
            if api_key:
                headers["x-api-key"] = api_key
                headers["anthropic-version"] = "2023-06-01"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.anthropic.com/v1/models", headers=headers,
                )
                resp.raise_for_status()
                return True, "Anthropic connected"
        except Exception as e:
            return False, f"Anthropic connection failed: {e}"

    if provider == "gemini":
        try:
            url = "https://generativelanguage.googleapis.com/v1/models"
            if api_key:
                url = f"{url}?key={api_key}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return True, "Gemini connected"
        except Exception as e:
            return False, f"Gemini connection failed: {e}"

    return False, f"Unknown provider: {provider}"


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
        # Safe default: cap output tokens to prevent CPU exhaustion on slow models.
        default_num_predict = 256
        if s.llm.mock:
            return LLMConfig(provider="mock", model="fake-list", request_timeout_seconds=timeout, num_predict=default_num_predict)
        provider = s.llm.provider
        if provider == "ollama":
            return LLMConfig(
                provider="ollama",
                model=s.llm.ollama.model,
                base_url=s.llm.ollama.base_url,
                request_timeout_seconds=timeout,
                num_predict=default_num_predict,
            )
        return LLMConfig(
            provider="gemini",
            model=s.llm.gemini.model,
            api_key=s.llm.gemini.api_key,
            request_timeout_seconds=timeout,
            num_predict=default_num_predict,
        )

    def _read_db(self) -> Optional[LLMConfig]:
        """Read the active llm_config row from the DB. Returns None if no row or error.

        Prefers the row with is_active=TRUE. Falls back to the latest by
        updated_at for backward compat with pre-migration databases.
        """
        try:
            pool = self._get_pool()
            if pool is None:
                return None
            with pool.connection() as conn:
                # Try the new schema: is_active column + request_timeout_seconds + num_predict + context_window_size.
                try:
                    row = conn.execute(
                        "SELECT provider, model, base_url, api_key, request_timeout_seconds, num_predict, context_window_size "
                        "FROM agent.llm_config WHERE is_active = TRUE LIMIT 1"
                    ).fetchone()
                    if not row:
                        # No active row — fall back to latest by updated_at.
                        row = conn.execute(
                            "SELECT provider, model, base_url, api_key, request_timeout_seconds, num_predict, context_window_size "
                            "FROM agent.llm_config ORDER BY updated_at DESC LIMIT 1"
                        ).fetchone()
                    if row:
                        return LLMConfig(
                            provider=row[0],
                            model=row[1],
                            base_url=row[2] or "",
                            api_key=row[3] or "",
                            request_timeout_seconds=row[4] or 300,
                            num_predict=row[5],
                            context_window_size=row[6] if len(row) > 6 else None,
                        )
                except Exception:
                    # Column doesn't exist yet (pre-migration) — fall back
                    row = conn.execute(
                        "SELECT provider, model, base_url, api_key, request_timeout_seconds, num_predict "
                        "FROM agent.llm_config ORDER BY updated_at DESC LIMIT 1"
                    ).fetchone()
                    if row:
                        return LLMConfig(
                            provider=row[0],
                            model=row[1],
                            base_url=row[2] or "",
                            api_key=row[3] or "",
                            request_timeout_seconds=row[4] or 300,
                            num_predict=row[5],
                        )
        except Exception as e:
            log.debug("Failed to read llm_config from DB: %s", e)
        return None

    def _build_llm(self, cfg: LLMConfig):
        """Build a chat model instance from config."""
        return build_llm(cfg, mock_responses=self._mock_responses)

    def _refresh(self):
        """Reload config from DB (or boot default) and rebuild the LLM.

        Skips the rebuild (and the INFO log) when the effective config has
        not changed since the last refresh — avoids log spam and unnecessary
        object reconstruction on every poll cycle.
        """
        cfg = self._read_db() or self._boot_default()

        # Skip rebuild if nothing changed (avoids log spam + object churn).
        with self._lock:
            if self._cfg == cfg and self._llm is not None:
                log.debug("LLM config unchanged: provider=%s model=%s", cfg.provider, cfg.model)
                return

        try:
            llm = self._build_llm(cfg)
        except Exception as e:
            log.warning("Failed to build LLM (provider=%s model=%s): %s — falling back to mock",
                        cfg.provider, cfg.model, e)
            cfg = LLMConfig(provider="mock", model="fallback", request_timeout_seconds=cfg.request_timeout_seconds)
            llm = self._build_llm(cfg)

        with self._lock:
            old_cfg = self._cfg
            self._cfg = cfg
            self._llm = llm

        # Only log at INFO on initial load or when the config actually changes.
        if old_cfg is None or old_cfg != cfg:
            log.info("LLM config loaded: provider=%s model=%s", cfg.provider, cfg.model)
        else:
            log.debug("LLM config unchanged: provider=%s model=%s", cfg.provider, cfg.model)

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


def get_llm_with_tools(llm, tool_defs: list[dict]):
    """Bind tool definitions to an LLM for native function calling.

    Returns llm.bind_tools(tool_defs) when the LLM supports it.
    Falls back to the plain LLM (no tool binding) when:
      - The LLM has no bind_tools method (e.g. FakeListChatModel for tests)
      - bind_tools() raises (model doesn't support tool calling)

    This allows the same graph code to work with both tool-capable models
    (Ollama with tools capability) and fallback models (text-based parsing).
    """
    bind = getattr(llm, "bind_tools", None)
    if bind is None:
        return llm
    try:
        return bind(tool_defs)
    except (TypeError, ValueError, NotImplementedError) as e:
        log.debug("bind_tools failed (%s) — falling back to plain LLM", e)
        return llm
