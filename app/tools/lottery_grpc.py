"""gRPC client to the Go lottery-stats-server.

Uses generated stubs from proto/lottery.proto (app/gen/lottery_pb2.py).
In dev/test mode where no Go service is running, the client can be
replaced with a mock via set_lottery_client().

Agent-facing API uses clear semantic names:
  - group_size (1=single, 2=pair, 3=triple, 4=quad, 5=quint, 6=full six-number group)
  - strength: "hot" (STRONG=2) or "cold" (WEAK=1)
  - window_from / window_to: optional date bounds (ISO 8601 strings or timestamps)

These map to the proto's form_type / strength / DateWindow fields internally.
The strong number is always separate from the six-number group.
"""

from __future__ import annotations

import functools
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

from app.config.settings import get_settings
from app.tools import ToolError

log = logging.getLogger(__name__)

_channel = None
_stub = None

# Strength mapping: agent-facing "hot"/"cold" → proto enum values.
# In the Go lottery-tree: Strong = frequent (hot), Weak = less frequent (cold).
_STRENGTH_MAP = {
    "hot": 2,   # Strength_STRONG
    "cold": 1,  # Strength_WEAK
    2: 2,
    1: 1,
}


# ── TTL result cache ──────────────────────────────────────────

class _ToolCache:
    """Thread-safe LRU cache with TTL for deterministic tool results.

    Cache key = (tool_name, canonical_args_json). Entries expire after ttl_seconds.
    When max_size is exceeded, the oldest entry is evicted (insertion-order LRU).

    TTL and max_size are read dynamically from the provided callables so that
    tests can monkeypatch them without recreating the cache singleton.
    """

    def __init__(self, ttl_fn: Callable[[], float], max_size_fn: Callable[[], int]):
        self._ttl_fn = ttl_fn
        self._max_size_fn = max_size_fn
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _make_key(tool: str, kwargs: dict) -> str:
        return f"{tool}:{json.dumps(kwargs, sort_keys=True, default=str)}"

    def get(self, tool: str, kwargs: dict) -> Any | None:
        key = self._make_key(tool, kwargs)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl_fn():
                del self._entries[key]
                return None
            # Move to end (most recently used).
            self._entries.move_to_end(key)
            return value

    def put(self, tool: str, kwargs: dict, value: Any) -> None:
        key = self._make_key(tool, kwargs)
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size_fn():
                self._entries.popitem(last=False)

    def invalidate(self) -> None:
        with self._lock:
            self._entries.clear()


_tool_cache: _ToolCache | None = None


def _get_cache() -> _ToolCache:
    """Lazily create the singleton tool cache from settings."""
    global _tool_cache
    if _tool_cache is None:
        # Use lambdas so monkeypatching _get_cache_ttl/_get_cache_max_size
        # in tests takes effect without recreating the cache singleton.
        _tool_cache = _ToolCache(
            ttl_fn=lambda: _get_cache_ttl(),
            max_size_fn=lambda: _get_cache_max_size(),
        )
    return _tool_cache


def _get_cache_ttl() -> float:
    return get_settings().tools.cache_ttl_seconds


def _get_cache_max_size() -> int:
    return get_settings().tools.cache_max_size


def invalidate_tool_cache() -> None:
    """Clear all cached tool results. Called after scraper runs (new data)."""
    if _tool_cache is not None:
        _tool_cache.invalidate()


def _cached(tool_name: str, kwargs: dict, fetch: Callable[[], dict]) -> dict:
    """Cache wrapper: check cache, call fetch on miss, store result.

    Errors (exceptions raised by fetch, or results containing an 'error'
    key for backward compat) are NOT cached — they may be transient
    (service unavailable) and should be retried on the next call.
    """
    cache = _get_cache()
    cached_result = cache.get(tool_name, kwargs)
    if cached_result is not None:
        return cached_result
    result = fetch()
    if isinstance(result, dict) and "error" not in result:
        cache.put(tool_name, kwargs, result)
    return result


def _get_stub():
    """Lazily create the gRPC stub."""
    global _channel, _stub
    if _stub is not None:
        return _stub

    s = get_settings()
    host = s.lottery_grpc.host
    port = s.lottery_grpc.port

    if not host:
        log.warning("LOTTERY_GRPC_HOST not set — lottery tools will return empty results")
        return None

    import grpc
    from app.gen import lottery_pb2_grpc
    _channel = grpc.insecure_channel(f"{host}:{port}")
    _stub = lottery_pb2_grpc.LotteryServiceStub(_channel)
    return _stub


def _build_window(window_from: str | None, window_to: str | None):
    """Build a proto DateWindow from optional ISO date strings.

    Returns None if both bounds are unset (full archive).
    """
    if not window_from and not window_to:
        return None
    from app.gen import lottery_pb2
    from google.protobuf.timestamp_pb2 import Timestamp
    import datetime

    window = lottery_pb2.DateWindow()
    if window_from:
        ts = Timestamp()
        # Accept ISO date or datetime strings.
        dt = datetime.datetime.fromisoformat(window_from.replace("Z", "+00:00"))
        ts.FromDatetime(dt)
        # 'from' is a Python keyword — use getattr to set the proto field.
        getattr(window, "from").CopyFrom(ts)
    if window_to:
        ts = Timestamp()
        dt = datetime.datetime.fromisoformat(window_to.replace("Z", "+00:00"))
        ts.FromDatetime(dt)
        window.to.CopyFrom(ts)
    return window


def generate_form(how_many: int, form_type: int, will_be: list[int] | None = None,
                  strength: int = 2, window_from: str | None = None,
                  window_to: str | None = None) -> dict:
    """Tool: generate lottery combinations via the Go lottery-stats-server.

    Args:
        how_many: number of forms to generate.
        form_type: systematic form size (default 6 = regular lotto).
        will_be: lucky numbers to include in every form.
        strength: 2=STRONG (frequent/hot), 1=WEAK (less frequent/cold).
        window_from / window_to: optional ISO date bounds for the historical window.
    """
    cache_kwargs = {
        "how_many": how_many, "form_type": form_type, "will_be": will_be,
        "strength": strength, "window_from": window_from, "window_to": window_to,
    }

    def _fetch() -> dict:
        if _mock_client is not None:
            fn = _mock_client.get("generate_form")
            if fn:
                return fn(how_many=how_many, form_type=form_type, will_be=will_be,
                          strength=strength, window_from=window_from, window_to=window_to)
            raise ToolError("Mock generate_form not configured")

        stub = _get_stub()
        if stub is None:
            raise ToolError("Lottery service unavailable")

        from app.gen import lottery_pb2
        window = _build_window(window_from, window_to)
        req = lottery_pb2.GenerateFormRequest(
            how_many=how_many,
            form_type=form_type,
            will_be=will_be or [],
            strength=strength,
        )
        if window is not None:
            req.window.CopyFrom(window)
        resp = stub.GenerateForm(req)
        forms = []
        for s in resp.forms:
            entry = {"numbers": list(s.numbers)}
            if s.HasField("strong"):
                entry["strong"] = s.strong
            forms.append(entry)
        return {"forms": forms}

    return _cached("generate_form", cache_kwargs, _fetch)


def get_statistics(how_many: int = 10, group_size: int = 2, strength: str | int = "hot",
                   window_from: str | None = None, window_to: str | None = None,
                   # Backward-compat aliases (old callers may pass form_type).
                   form_type: int | None = None) -> dict:
    """Tool: calculate frequent/infrequent number groups via the Go lottery-stats-server.

    Args:
        how_many: number of groups to return (top-N by frequency).
        group_size: 1=single number, 2=pair, 3=triple, 4=quad, 5=quint,
                    6=full six-number group (excluding strong number).
        strength: "hot" (most frequent, STRONG) or "cold" (least frequent, WEAK).
                  Also accepts proto enum ints: 2=hot, 1=cold.
        window_from / window_to: optional ISO date bounds for the historical window.
        form_type: deprecated alias for group_size (backward compat).

    Returns:
        {"groups": [{"numbers": [..], "count": N}, ...]}
    """
    # Resolve backward-compat alias.
    if form_type is not None and group_size == 2:
        group_size = form_type
    # Map strength to proto enum value.
    strength_val = _STRENGTH_MAP.get(strength, 2) if not isinstance(strength, int) else strength

    cache_kwargs = {
        "how_many": how_many, "group_size": group_size, "strength": strength_val,
        "window_from": window_from, "window_to": window_to,
    }

    def _fetch() -> dict:
        if _mock_client is not None:
            fn = _mock_client.get("get_statistics")
            if fn:
                return fn(how_many=how_many, group_size=group_size, strength=strength_val,
                          window_from=window_from, window_to=window_to)
            raise ToolError("Mock get_statistics not configured")

        stub = _get_stub()
        if stub is None:
            raise ToolError("Lottery service unavailable")

        from app.gen import lottery_pb2
        window = _build_window(window_from, window_to)
        req = lottery_pb2.GetStatisticsRequest(
            how_many=how_many,
            form_type=group_size,
            strength=strength_val,
        )
        if window is not None:
            req.window.CopyFrom(window)
        resp = stub.GetStatistics(req)
        return {"groups": [{"numbers": list(p.numbers), "count": p.count} for p in resp.pairs]}

    return _cached("get_statistics", cache_kwargs, _fetch)


def analyze(form: list[int], window_from: str | None = None,
            window_to: str | None = None) -> dict:
    """Tool: analyze user-selected numbers against historical draws.

    Returns frequency groups for all subset sizes 1-6 of the selected numbers,
    plus the archive size used for the analysis.

    Args:
        form: the user's selected numbers (1-6 regular numbers).
              If a 7th number (strong) is included, it is stripped by the Go service.
        window_from / window_to: optional ISO date bounds for the historical window.

    Returns:
        {
            "frequency_groups": [
                {"size": 1, "combos": C(37,1), "entries": [{"numbers":[n], "count": c}, ...]},
                {"size": 2, "combos": C(37,2), "entries": [{"numbers":[a,b], "count": c}, ...]},
                ...
                {"size": 6, "combos": C(37,6), "entries": [{"numbers":[...6...], "count": c}, ...]},
            ],
            "archive_size": N
        }
    """
    cache_kwargs = {"form": form, "window_from": window_from, "window_to": window_to}

    def _fetch() -> dict:
        if _mock_client is not None:
            fn = _mock_client.get("analyze")
            if fn:
                return fn(form=form, window_from=window_from, window_to=window_to)
            raise ToolError("Mock analyze not configured")

        stub = _get_stub()
        if stub is None:
            raise ToolError("Lottery service unavailable")

        from app.gen import lottery_pb2
        window = _build_window(window_from, window_to)
        req = lottery_pb2.AnalyzeRequest(form=form)
        if window is not None:
            req.window.CopyFrom(window)
        resp = stub.Analyze(req)
        return {
            "frequency_groups": [
                {
                    "size": g.size,
                    "combos": g.combos,
                    "entries": [
                        {"numbers": list(e.numbers), "count": e.count}
                        for e in g.entries
                    ],
                }
                for g in resp.frequency_groups
            ],
            "archive_size": resp.archive_size,
        }

    return _cached("analyze", cache_kwargs, _fetch)


def simulate(form: list[int], strong: int = 0,
             archive_from: str | None = None, archive_to: str | None = None,
             simulate_from: str | None = None, simulate_to: str | None = None,
             ticket_cost: float = 3.0, prize_amounts: list[float] | None = None) -> dict:
    """Tool: backtest user-selected numbers against historical draws.

    Supports systematic forms (6, 8, 10, 12 numbers) where all C(N,6)
    combinations are played per draw.

    Args:
        form: the user's selected numbers (6, 8, 10, or 12 for systematic forms).
        strong: the user's strong number (1-7). 0 = no strong number.
        archive_from / archive_to: optional ISO date bounds for the archive window
            (the historical-draw range loaded as context). Unset = full archive.
        simulate_from / simulate_to: optional ISO date bounds for the simulate
            window (the sub-range of draws to backtest against). Unset = falls
            back to archive_window (legacy single-window behavior).
        ticket_cost: ticket cost per table/combination in ILS (default 3.0).
        prize_amounts: optional user-configurable prize amounts per tier (ILS).
            Index 0 = tier 1 (6+strong), ..., 7 = tier 8 (3). Must be length 0 or 8.

    Returns:
        {"draws": [...], "summary": {total_draws, total_spent, total_won, net, ...}}
    """
    cache_kwargs = {
        "form": form, "strong": strong,
        "archive_from": archive_from, "archive_to": archive_to,
        "simulate_from": simulate_from, "simulate_to": simulate_to,
        "ticket_cost": ticket_cost, "prize_amounts": prize_amounts,
    }

    def _fetch() -> dict:
        if _mock_client is not None:
            fn = _mock_client.get("simulate")
            if fn:
                return fn(form=form, strong=strong,
                          archive_from=archive_from, archive_to=archive_to,
                          simulate_from=simulate_from, simulate_to=simulate_to,
                          ticket_cost=ticket_cost, prize_amounts=prize_amounts)
            raise ToolError("Mock simulate not configured")

        stub = _get_stub()
        if stub is None:
            raise ToolError("Lottery service unavailable")

        from app.gen import lottery_pb2
        req = lottery_pb2.SimulateRequest(
            form=form,
            strong=strong,
            ticket_cost=ticket_cost,
        )
        archive_window = _build_window(archive_from, archive_to)
        if archive_window is not None:
            req.archive_window.CopyFrom(archive_window)
        simulate_window = _build_window(simulate_from, simulate_to)
        if simulate_window is not None:
            req.simulate_window.CopyFrom(simulate_window)
        if prize_amounts:
            req.prize_amounts.extend(prize_amounts)
        resp = stub.Simulate(req)

        draws = []
        for d in resp.draws:
            tier_hits = []
            for h in d.tier_hits:
                tier_hits.append({
                    "tier": h.tier,
                    "hits": h.hits,
                    "amount_per_hit": h.amount_per_hit,
                    "total": h.total,
                })
            draws.append({
                "draw_number": d.draw_number,
                "winning_numbers": list(d.winning_numbers),
                "winning_strong": d.winning_strong,
                "tier_hits": tier_hits,
                "prize_won": d.prize_won,
                "ticket_cost": d.ticket_cost,
                "used_real_prizes": d.used_real_prizes,
            })

        summary = None
        if resp.HasField("summary"):
            s = resp.summary
            tier_summaries = []
            for ts in s.tier_summaries:
                tier_summaries.append({
                    "tier": ts.tier,
                    "label": ts.label,
                    "total_hits": ts.total_hits,
                    "total_amount": ts.total_amount,
                })
            summary = {
                "total_draws": s.total_draws,
                "total_combinations": s.total_combinations,
                "total_spent": s.total_spent,
                "total_won": s.total_won,
                "net": s.net,
                "tier_summaries": tier_summaries,
                "draws_with_real_prizes": s.draws_with_real_prizes,
            }

        return {"draws": draws, "summary": summary}

    return _cached("simulate", cache_kwargs, _fetch)


# ── Mock support for testing ─────────────────────────────────

_mock_client: Optional[dict] = None


def set_mock_client(mock: dict):
    """Inject a mock lottery client for testing.

    The mock is a dict of function name -> callable.
    Example: {"generate_form": lambda **kw: {"forms": [[1,2,3]]}}
    """
    global _mock_client
    _mock_client = mock


def reset_mock_client():
    global _mock_client
    _mock_client = None
