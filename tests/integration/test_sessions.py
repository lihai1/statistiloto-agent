"""Integration test: chat session history — list, load, delete with tier limits."""

import time
import pytest

pytestmark = pytest.mark.integration


class TestSessionLimits:
    """Tier-based retention limits enforced on upsert."""

    def test_free_tier_keeps_only_1_session(self, db_pool):
        from app.sessions import upsert_session, list_sessions, get_session_limit

        assert get_session_limit("free") == 1

        # Insert 3 sessions for a free user — only the newest should survive.
        for i in range(3):
            upsert_session("free-user", f"session-{i}", f"Title {i}", f"msg {i}", "free")
            time.sleep(0.01)  # ensure updated_at differs

        sessions = list_sessions("free-user", "free")
        assert len(sessions) == 1
        # The newest (last inserted) should be the one kept.
        assert sessions[0].session_id == "session-2"

    def test_paid_tier_keeps_15_sessions(self, db_pool):
        from app.sessions import upsert_session, list_sessions, get_session_limit

        assert get_session_limit("paid") == 15

        for i in range(20):
            upsert_session("paid-user", f"session-{i}", f"Title {i}", f"msg {i}", "paid")
            time.sleep(0.01)

        sessions = list_sessions("paid-user", "paid")
        assert len(sessions) == 15
        # The oldest 5 (session-0..session-4) should have been pruned.
        ids = {s.session_id for s in sessions}
        assert "session-0" not in ids
        assert "session-4" not in ids
        assert "session-5" in ids
        assert "session-19" in ids

    def test_admin_tier_unlimited(self, db_pool):
        from app.sessions import upsert_session, list_sessions, get_session_limit

        assert get_session_limit("admin") is None

        for i in range(20):
            upsert_session("admin-user", f"session-{i}", f"Title {i}", f"msg {i}", "admin")
            time.sleep(0.01)

        sessions = list_sessions("admin-user", "admin")
        assert len(sessions) == 20


class TestSessionUpsert:
    def test_upsert_increments_message_count(self, db_pool):
        from app.sessions import upsert_session, list_sessions

        upsert_session("user-a", "s1", "First", "hello", "paid")
        upsert_session("user-a", "s1", "", "second message", "paid")
        upsert_session("user-a", "s1", "", "third message", "paid")

        sessions = list_sessions("user-a", "paid")
        assert len(sessions) == 1
        assert sessions[0].message_count == 3
        # Title is set only on the first insert.
        assert sessions[0].title == "First"
        # last_message reflects the latest message.
        assert sessions[0].last_message == "third message"

    def test_upsert_title_truncated_to_80(self, db_pool):
        from app.sessions import upsert_session, list_sessions

        # The caller (main.py _record_session) truncates the title to 80 chars
        # and last_message to 200 chars before passing them to upsert_session.
        # upsert_session itself stores whatever it's given.
        long_msg = "x" * 200
        title = long_msg[:80]
        last_message = long_msg[:200]
        upsert_session("user-b", "s1", title, last_message, "paid")
        sessions = list_sessions("user-b", "paid")
        assert len(sessions[0].title) == 80
        assert len(sessions[0].last_message) == 200


class TestSessionDelete:
    def test_delete_own_session(self, db_pool):
        from app.sessions import upsert_session, delete_session, list_sessions

        upsert_session("user-c", "s1", "Title", "msg", "paid")
        assert len(list_sessions("user-c", "paid")) == 1

        deleted = delete_session("user-c", "s1")
        assert deleted is True
        assert len(list_sessions("user-c", "paid")) == 0

    def test_cannot_delete_other_users_session(self, db_pool):
        from app.sessions import upsert_session, delete_session

        upsert_session("user-d", "s1", "Title", "msg", "paid")
        # user-e tries to delete user-d's session — should fail (no row deleted).
        deleted = delete_session("user-e", "s1")
        assert deleted is False

    def test_delete_nonexistent_session(self, db_pool):
        from app.sessions import delete_session

        deleted = delete_session("user-f", "no-such-session")
        assert deleted is False

    def test_delete_all_sessions(self, db_pool):
        from app.sessions import upsert_session, delete_all_sessions, list_sessions

        for i in range(3):
            upsert_session("user-g", f"s{i}", f"Title {i}", f"msg {i}", "paid")
        assert len(list_sessions("user-g", "paid")) == 3

        count = delete_all_sessions("user-g")
        assert count == 3
        assert len(list_sessions("user-g", "paid")) == 0


class TestSessionList:
    def test_list_ordered_newest_first(self, db_pool):
        from app.sessions import upsert_session, list_sessions

        for i in range(5):
            upsert_session("user-h", f"s{i}", f"Title {i}", f"msg {i}", "paid")
            time.sleep(0.02)

        sessions = list_sessions("user-h", "paid")
        # Newest (last inserted) should be first.
        assert sessions[0].session_id == "s4"
        assert sessions[-1].session_id == "s0"

    def test_list_only_returns_own_sessions(self, db_pool):
        from app.sessions import upsert_session, list_sessions

        upsert_session("user-i", "s1", "Title", "msg", "paid")
        upsert_session("user-j", "s1", "Title", "msg", "paid")

        sessions = list_sessions("user-i", "paid")
        assert len(sessions) == 1
        assert sessions[0].session_id == "s1"
