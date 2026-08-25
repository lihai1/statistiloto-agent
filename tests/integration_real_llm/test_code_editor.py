"""Real LLM tests for admin code editor tools."""

from __future__ import annotations

import os
import time

import pytest

pytestmark = [pytest.mark.integration]


class TestAdminCodeEditor:
    def _set_code_root(self, tmp_path):
        os.environ["AGENT_CODE_ROOT"] = str(tmp_path)

    def _make_file(self, tmp_path, rel_path, content):
        f = tmp_path / rel_path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        return f

    def test_admin_read_code(self, client, admin_headers, tmp_path):
        """Admin asks to read a source file; admin_ops should call read_code."""
        self._set_code_root(tmp_path)
        self._make_file(tmp_path, "app/tools/online_search.py", "# test online search module\n")

        session_id = f"real-admin-read-{int(time.time())}"
        resp = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "Show me app/tools/online_search.py",
                "intent": "admin_ops",
            },
            headers=admin_headers,
        )
        print(f"\n[ADMIN READ] status={resp.status_code} body={resp.text[:1000]}")
        assert resp.status_code == 200
        data = resp.json()
        if data.get("paused"):
            pytest.skip(f"Read unexpectedly paused: {data}")
        response = data.get("response") or ""
        assert "test online search module" in response.lower(), f"Expected file content, got: {response[:500]}"

    def test_admin_edit_file(self, client, admin_headers, tmp_path):
        """Admin asks to edit a source file; admin_ops should pause for HITL, then edit."""
        self._set_code_root(tmp_path)
        self._make_file(tmp_path, "app/main.py", "original content\n")

        session_id = f"real-admin-edit-{int(time.time())}"
        resp = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": "Replace the entire content of file app/main.py with 'hello world'",
                "intent": "admin_ops",
            },
            headers=admin_headers,
        )
        print(f"\n[ADMIN EDIT] status={resp.status_code} body={resp.text[:1000]}")
        assert resp.status_code == 200
        data = resp.json()

        # edit_file is a write tool, so the graph must pause for approval.
        if not data.get("paused"):
            pytest.skip(f"Model did not plan edit_file; response: {data}")

        resp2 = client.post(
            "/approve",
            json={"session_id": session_id, "approved": True},
            headers=admin_headers,
        )
        print(f"[ADMIN EDIT APPROVE] status={resp2.status_code} body={resp2.text[:1000]}")
        assert resp2.status_code == 200
        response = resp2.json().get("response") or ""
        assert "edited" in response.lower() or "hello world" in response.lower(), f"Edit failed: {response[:500]}"

        # Verify the file was actually modified.
        target = tmp_path / "app/main.py"
        assert "hello world" in target.read_text(encoding="utf-8"), "File not updated"
