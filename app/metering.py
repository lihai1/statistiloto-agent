"""Token metering — wraps every LLM call and logs to agent.token_usage.

Also enforces per-user daily budgets.
"""

from __future__ import annotations

import functools
import logging
import time

from app.config.settings import get_settings, get_tier_config
from app.llm.config_store import get_llm_store
from app.rag.store import get_pool

log = logging.getLogger(__name__)

# Rough cost estimates per 1K tokens (USD). Ollama is local/free.
_COST_PER_1K = {
    "ollama": {"prompt": 0.0, "completion": 0.0},
    "gemini": {"prompt": 0.000075, "completion": 0.0003},  # gemini-2.0-flash approx
    "mock": {"prompt": 0.0, "completion": 0.0},
}


def _cost_usd(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _COST_PER_1K.get(provider, _COST_PER_1K["mock"])
    return (prompt_tokens / 1000.0 * rates["prompt"]) + (completion_tokens / 1000.0 * rates["completion"])


def meter_llm(fn):
    """Decorator: wrap an LLM-calling node function to log token usage.

    The wrapped node function should include '_usage' (LangChain UsageMetadata
    dict with input_tokens/output_tokens) and optionally '_response_metadata'
    (raw provider metadata, e.g. Ollama's prompt_eval_count/eval_count) in
    its return dict so the decorator can extract token counts.
    """
    @functools.wraps(fn)
    def wrapper(state, *a, **kw):
        result = fn(state, *a, **kw)

        # Extract token usage from the node's return dict.
        # Node functions return a dict (LangGraph state update), not the
        # raw AIMessage, so we look for _usage / _response_metadata keys
        # that the node function populates from the LLM response.
        prompt_tokens = 0
        completion_tokens = 0

        if isinstance(result, dict):
            # Primary: LangChain normalized usage_metadata
            um = result.get("_usage")
            if um:
                prompt_tokens = um.get("input_tokens", 0)
                completion_tokens = um.get("output_tokens", 0)
            else:
                # Fallback: raw provider response_metadata
                # Ollama: prompt_eval_count (input), eval_count (output)
                rm = result.get("_response_metadata")
                if rm:
                    prompt_tokens = rm.get("prompt_eval_count", 0) or rm.get("input_tokens", 0)
                    completion_tokens = rm.get("eval_count", 0) or rm.get("output_tokens", 0)
        elif hasattr(result, "usage_metadata") and result.usage_metadata:
            # Direct AIMessage (not currently used, but kept for safety)
            prompt_tokens = result.usage_metadata.get("input_tokens", 0)
            completion_tokens = result.usage_metadata.get("output_tokens", 0)

        # Get provider/model from the global config store.
        cfg = get_llm_store().get_config()
        cost = _cost_usd(cfg.provider, cfg.model, prompt_tokens, completion_tokens)

        user_sub = state.get("user_sub", "unknown")
        tier = state.get("tier", "free")
        session_id = state.get("session_id", "")
        thread_id = f"{user_sub}:{session_id}"

        try:
            pool = get_pool()
            with pool.connection() as conn:
                conn.execute(
                    """INSERT INTO agent.token_usage
                       (thread_id, user_sub, tier, provider, model,
                        prompt_tokens, completion_tokens, cost_usd, ts)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (thread_id, user_sub, tier, cfg.provider, cfg.model,
                     prompt_tokens, completion_tokens, cost, time.time()),
                )
        except Exception as e:
            log.warning("Failed to log token usage: %s", e)

        # Strip the temporary metadata keys so they don't pollute LangGraph state.
        if isinstance(result, dict):
            result.pop("_usage", None)
            result.pop("_response_metadata", None)

        return result

    return wrapper


def check_daily_budget(user_sub: str, tier: str) -> bool:
    """Check if the user is within their daily token budget. Returns True if OK."""
    cfg = get_tier_config(tier)
    if cfg.daily_budget_usd <= 0:
        return True

    try:
        pool = get_pool()
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM agent.token_usage "
                "WHERE user_sub = %s AND ts > extract(epoch from now() - interval '24 hours')",
                (user_sub,),
            ).fetchone()
        spent = float(row[0]) if row else 0.0
        return spent < cfg.daily_budget_usd
    except Exception as e:
        log.warning("Failed to check daily budget: %s", e)
        return True  # fail open
