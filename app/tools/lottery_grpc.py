"""gRPC client to the Go lottery-stats-server.

Uses generated stubs from proto/lottery.proto (app/gen/lottery_pb2.py).
In dev/test mode where no Go service is running, the client can be
replaced with a mock via set_lottery_client().
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config.settings import get_settings

log = logging.getLogger(__name__)

_channel = None
_stub = None


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


def generate_form(how_many: int, form_type: int, will_be: list[int] | None = None,
                  strength: int = 2) -> dict:
    """Tool: generate lottery combinations via the Go lottery-stats-server."""
    if _mock_client is not None:
        fn = _mock_client.get("generate_form")
        if fn:
            return fn(how_many=how_many, form_type=form_type, will_be=will_be, strength=strength)
        return {"forms": [], "error": "Mock generate_form not configured"}

    stub = _get_stub()
    if stub is None:
        return {"forms": [], "error": "Lottery service unavailable"}

    from app.gen import lottery_pb2
    resp = stub.GenerateForm(lottery_pb2.GenerateFormRequest(
        how_many=how_many,
        form_type=form_type,
        will_be=will_be or [],
        strength=strength,
    ))
    return {"forms": [[n for n in s.numbers] for s in resp.forms]}


def get_statistics(how_many: int, form_type: int, strength: int = 2) -> dict:
    """Tool: calculate frequent pairs/groups via the Go lottery-stats-server."""
    if _mock_client is not None:
        fn = _mock_client.get("get_statistics")
        if fn:
            return fn(how_many=how_many, form_type=form_type, strength=strength)
        return {"pairs": [], "error": "Mock get_statistics not configured"}

    stub = _get_stub()
    if stub is None:
        return {"pairs": [], "error": "Lottery service unavailable"}

    from app.gen import lottery_pb2
    resp = stub.GetStatistics(lottery_pb2.GetStatisticsRequest(
        how_many=how_many,
        form_type=form_type,
        strength=strength,
    ))
    return {"pairs": [{"numbers": list(p.numbers), "count": p.count} for p in resp.pairs]}


def analyze(form: list[int]) -> dict:
    """Tool: analyze user-selected numbers against historical draws."""
    if _mock_client is not None:
        fn = _mock_client.get("analyze")
        if fn:
            return fn(form=form)
        return {"matches": [], "frequency": {}, "error": "Mock analyze not configured"}

    stub = _get_stub()
    if stub is None:
        return {"matches": [], "frequency": {}, "error": "Lottery service unavailable"}

    from app.gen import lottery_pb2
    resp = stub.Analyze(lottery_pb2.AnalyzeRequest(form=form))
    return {
        "matches": [
            {
                "draw_id": m.draw_id,
                "draw_date": m.draw_date,
                "matched_numbers": list(m.matched_numbers),
                "match_count": m.match_count,
            }
            for m in resp.matches
        ],
        "frequency": {str(k): v for k, v in resp.frequency.items()},
    }


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
