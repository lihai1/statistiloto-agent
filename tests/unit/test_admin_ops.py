import pytest
from unittest.mock import MagicMock, patch
from app.security import TokenClaims
from app.tools.admin_ops import trigger_scraper


def test_trigger_scraper_requires_admin():
    user_claims = TokenClaims(
        sub="user-1",
        tier="free",
        roles=["USER"],
        raw_token="fake-token",
    )
    with pytest.raises(Exception):
        trigger_scraper(user_claims)


def test_trigger_scraper_when_redis_unavailable():
    admin_claims = TokenClaims(
        sub="admin-1",
        tier="admin",
        roles=["USER", "ADMIN"],
        raw_token="fake-admin-token",
    )
    with patch("app.tools.admin_ops.get_sync_redis", return_value=None):
        res = trigger_scraper(admin_claims)
        assert res["status"] == "scraper_unavailable"
        assert res["error"] == "redis_unavailable"


def test_trigger_scraper_enqueues_and_reads_done_event():
    admin_claims = TokenClaims(
        sub="admin-1",
        tier="admin",
        roles=["USER", "ADMIN"],
        raw_token="fake-admin-token",
    )
    mock_redis = MagicMock()
    # xread returns a done event
    mock_redis.xread.return_value = [
        ("scraper:events:test", [
            ("1-0", {"event": "progress", "phase": "fetch"}),
            ("2-0", {"event": "done", "inserted": "3", "prizes_written": "6"}),
        ])
    ]

    with patch("app.tools.admin_ops.get_sync_redis", return_value=mock_redis), \
         patch("app.tools.admin_ops._audit"):
        res = trigger_scraper(admin_claims)
        assert res["status"] == "scraper_done"
        assert res["inserted"] == 3
        assert res["prizes_written"] == 6

        assert mock_redis.xadd.called
        assert mock_redis.hset.called
        assert mock_redis.set.called


def test_trigger_scraper_status_hash_fallback():
    admin_claims = TokenClaims(
        sub="admin-1",
        tier="admin",
        roles=["USER", "ADMIN"],
        raw_token="fake-admin-token",
    )
    mock_redis = MagicMock()
    # xread returns nothing
    mock_redis.xread.return_value = []
    # hgetall returns done
    mock_redis.hgetall.return_value = {
        "status": "done",
        "inserted": "4",
        "prizes_written": "8",
    }

    with patch("app.tools.admin_ops.get_sync_redis", return_value=mock_redis), \
         patch("app.tools.admin_ops._audit"):
        res = trigger_scraper(admin_claims)
        assert res["status"] == "scraper_done"
        assert res["inserted"] == 4
        assert res["prizes_written"] == 8


def test_trigger_scraper_error_event():
    admin_claims = TokenClaims(
        sub="admin-1",
        tier="admin",
        roles=["USER", "ADMIN"],
        raw_token="fake-admin-token",
    )
    mock_redis = MagicMock()
    mock_redis.xread.return_value = [
        ("scraper:events:test", [
            ("1-0", {"event": "error", "message": "Pais scraper blocked"}),
        ])
    ]

    with patch("app.tools.admin_ops.get_sync_redis", return_value=mock_redis), \
         patch("app.tools.admin_ops._audit"):
        res = trigger_scraper(admin_claims)
        assert res["status"] == "scraper_failed"
        assert "Pais scraper blocked" in res["error"]
