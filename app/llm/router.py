"""LLM router — thin wrapper around config_store for get_llm().

Kept as a separate module so graphs can import `from app.llm.router import get_llm`
without depending on config_store internals.
"""

from app.llm.config_store import get_llm, get_llm_store, LLMConfig

__all__ = ["get_llm", "get_llm_store", "LLMConfig"]
