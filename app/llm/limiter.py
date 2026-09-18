"""LLM call serializer — process-wide lock so only one LLM inference runs at a time.

Ollama is constrained to ``OLLAMA_NUM_PARALLEL=1`` (one in-flight inference per
model). When two users call the chat agent concurrently, both ``graph.invoke()``
runs hit Ollama simultaneously; the second request can fail or time out
non-gracefully. This module centralizes serialization so all worker subgraphs
share a single queue — the second call waits for the first to complete, then
proceeds naturally.

The lock is ``threading.Lock`` because ``graph.invoke()`` runs synchronously in
``asyncio.to_thread`` — each graph execution occupies a separate thread.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger(__name__)

# Process-wide lock — single in-flight LLM inference across all threads.
_llm_lock = threading.Lock()


def llm_invoke(llm: Any, prompt: Any, **kwargs: Any) -> Any:
    """Call ``llm.invoke(prompt)`` under the global LLM lock.

    Acquires the process-wide lock so only one LLM inference runs at a time.
    The caller's thread blocks until the lock is released by the prior call,
    then proceeds. This makes Ollama's single-inference constraint transparent
    to concurrent users — the second user simply waits for the first to finish.

    Args:
        llm: the LLM instance (or tool-bound variant) to invoke.
        prompt: the prompt to pass to ``llm.invoke()``.
        **kwargs: additional keyword arguments forwarded to ``invoke()``.

    Returns:
        The raw LLM response (e.g. ``AIMessage``).
    """
    with _llm_lock:
        log.debug("[llm-limiter] ACQUIRED — calling llm.invoke()")
        return llm.invoke(prompt, **kwargs)
