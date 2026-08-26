"""LLM router — thin wrapper around config_store for get_llm().

Kept as a separate module so graphs can import `from app.llm.router import get_llm`
without depending on config_store internals.

Supports per-request LLM overrides for admin users via a ContextVar.
When set (via set_llm_override / reset_llm_override), the override LLM
is returned instead of the global store LLM. The ContextVar propagates
into threads spawned by asyncio.to_thread because to_thread copies the
current context.
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

from app.llm.config_store import get_llm as _get_global_llm, get_llm_store, LLMConfig

__all__ = ["get_llm", "get_llm_store", "LLMConfig", "set_llm_override", "reset_llm_override"]

# ContextVar: when non-None, get_llm() returns this instead of the global LLM.
# Propagates into asyncio.to_thread threads (which copy the current context).
_llm_override: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "llm_override", default=None
)


def get_llm():
    """Return the LLM for the current request context.

    If an admin override is active (set via set_llm_override), returns that.
    Otherwise returns the single global LLM from the config store.
    """
    override = _llm_override.get()
    if override is not None:
        return override
    return _get_global_llm()


def set_llm_override(llm: Any) -> contextvars.Token:
    """Set a per-request LLM override. Returns a token for reset_llm_override."""
    return _llm_override.set(llm)


def reset_llm_override(token: contextvars.Token) -> None:
    """Clear the per-request LLM override using the token from set_llm_override."""
    _llm_override.reset(token)
