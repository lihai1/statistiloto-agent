"""LLM call serializer — per-provider lock so constrained backends run one
inference at a time.

Ollama is constrained to ``OLLAMA_NUM_PARALLEL=1`` (one in-flight inference per
model). When two users call the chat agent concurrently, both graph runs hit
Ollama simultaneously; the second request can fail or time out non-gracefully.
This module serializes calls — but ONLY for providers that need it. Cloud
providers (gemini/openai/anthropic/mock) handle their own concurrency, so
they bypass the lock entirely instead of queueing behind a local inference.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

# Lock — single in-flight LLM inference, only acquired for constrained providers.
_llm_lock = threading.Lock()

# Providers constrained to one in-flight inference. "mock" is the test
# provider — serializing it keeps the concurrency invariant verifiable.
_SERIALIZED_PROVIDERS = {"ollama", "mock"}


def _needs_serialization() -> bool:
    """True when the active provider requires single-inference serialization."""
    try:
        from app.llm.config_store import get_llm_store
        return get_llm_store().get_config().provider in _SERIALIZED_PROVIDERS
    except Exception:
        return True  # fail safe: serialize when the store is unavailable


def llm_invoke(llm: Any, prompt: Any, **kwargs: Any) -> Any:
    """Call ``llm.invoke(prompt)`` under the provider lock when needed.

    Ollama calls serialize so only one inference runs at a time; other
    providers run concurrently. Additional kwargs forward to ``invoke()``.
    """
    if _needs_serialization():
        with _llm_lock:
            log.debug("[llm-limiter] ACQUIRED — calling llm.invoke()")
            return llm.invoke(prompt, **kwargs)
    return llm.invoke(prompt, **kwargs)


def llm_stream_invoke(llm: Any, prompt: Any, **kwargs: Any) -> Any:
    """Stream ``llm.stream(prompt)`` and accumulate chunks into one message.

    Uses ``.stream()`` so LangGraph's ``stream_mode="messages"`` emits real
    token chunks to stream consumers, then folds the chunks back into a
    single accumulated message (same ``.content`` / ``usage_metadata`` shape
    as an invoke result) for the node's normal handling.
    """
    acc = None
    if _needs_serialization():
        with _llm_lock:
            for chunk in llm.stream(prompt, **kwargs):
                acc = chunk if acc is None else acc + chunk
    else:
        for chunk in llm.stream(prompt, **kwargs):
            acc = chunk if acc is None else acc + chunk
    return acc
