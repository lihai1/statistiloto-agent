"""Free-tier LLM toggle — controls whether free users can invoke the LLM.

Default: disabled (free users receive a generic deterministic response for
ambiguous/domain-explanation requests instead of calling the LLM).

The initial value is read from settings (env var ``FREE_LLM_ENABLED`` or
``agent.yaml``). An admin can flip it at runtime via the ``PUT /free-llm``
API endpoint. The toggle is process-scoped (in-memory) — it does NOT persist
across restarts. Set the env var for persistent defaults.

Security: this toggle only controls whether free-tier ambiguous/domain
requests reach the nl_assistant worker. It does NOT grant free users any
additional tools, capabilities, or write access. Authorization is still
enforced by CapabilityConfig using trusted JWT claims.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_lock = threading.Lock()
_enabled: bool | None = None  # None = not yet initialized from settings


def is_free_llm_enabled() -> bool:
    """Return True if free-tier users are allowed to invoke the LLM.

    Reads the initial value lazily from settings on first call.
    """
    global _enabled
    with _lock:
        if _enabled is None:
            from app.config.settings import get_settings
            _enabled = get_settings().free_tier.llm_enabled
            log.info("[free_tier_llm] initialized from settings: enabled=%s", _enabled)
        return _enabled


def set_free_llm_enabled(enabled: bool) -> None:
    """Set the free-tier LLM toggle at runtime (admin API)."""
    global _enabled
    with _lock:
        old = _enabled
        _enabled = bool(enabled)
        log.info("[free_tier_llm] toggle changed: %s -> %s", old, _enabled)


def reset_free_llm_toggle() -> None:
    """Reset the toggle to uninitialized state (for tests)."""
    global _enabled
    with _lock:
        _enabled = None
