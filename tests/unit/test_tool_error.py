"""Unit tests for tool error propagation (#7) — no DB needed."""

import os

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

import pytest

from app.tools import ToolError
from app.tools import code_editor, lottery_grpc


class TestToolError:
    """Tests for the ToolError exception class."""

    def test_tool_error_is_exception(self):
        assert issubclass(ToolError, Exception)

    def test_tool_error_message(self):
        err = ToolError("something went wrong")
        assert str(err) == "something went wrong"

    def test_tool_error_can_be_raised(self):
        with pytest.raises(ToolError, match="fail"):
            raise ToolError("fail")


class TestCodeEditorErrorPropagation:
    """Tests that code_editor raises ToolError instead of returning error dicts."""

    def test_read_code_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_CODE_ROOT", str(tmp_path))
        with pytest.raises(ToolError, match="File not found"):
            code_editor.read_code("nonexistent.py")

    def test_read_code_directory_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_CODE_ROOT", str(tmp_path))
        (tmp_path / "subdir").mkdir()
        with pytest.raises(ToolError, match="directory"):
            code_editor.read_code("subdir")

    def test_list_files_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_CODE_ROOT", str(tmp_path))
        with pytest.raises(ToolError, match="Directory not found"):
            code_editor.list_files("nonexistent")

    def test_edit_file_missing_args_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_CODE_ROOT", str(tmp_path))
        with pytest.raises(ToolError, match="content.*old_string"):
            code_editor.edit_file("some.py")

    def test_edit_file_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_CODE_ROOT", str(tmp_path))
        with pytest.raises(ToolError, match="File not found"):
            code_editor.edit_file("nonexistent.py", content="hello")

    def test_edit_file_old_string_not_found_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_CODE_ROOT", str(tmp_path))
        test_file = tmp_path / "test.py"
        test_file.write_text("hello world")
        with pytest.raises(ToolError, match="old_string not found"):
            code_editor.edit_file("test.py", old_string="xyz", new_string="abc")


class TestLotteryGrpcErrorPropagation:
    """Tests that lottery_grpc raises ToolError instead of returning error dicts."""

    def test_generate_form_service_unavailable_raises(self, monkeypatch):
        # Ensure no mock client and no gRPC host.
        lottery_grpc.reset_mock_client()
        monkeypatch.setattr("app.tools.lottery_grpc._get_stub", lambda: None)
        # Clear cache to force a fresh fetch.
        lottery_grpc.invalidate_tool_cache()
        with pytest.raises(ToolError, match="Lottery service unavailable"):
            lottery_grpc.generate_form(how_many=1, form_type=6)

    def test_get_statistics_service_unavailable_raises(self, monkeypatch):
        lottery_grpc.reset_mock_client()
        monkeypatch.setattr("app.tools.lottery_grpc._get_stub", lambda: None)
        lottery_grpc.invalidate_tool_cache()
        with pytest.raises(ToolError, match="Lottery service unavailable"):
            lottery_grpc.get_statistics(how_many=10, group_size=2)

    def test_analyze_service_unavailable_raises(self, monkeypatch):
        lottery_grpc.reset_mock_client()
        monkeypatch.setattr("app.tools.lottery_grpc._get_stub", lambda: None)
        lottery_grpc.invalidate_tool_cache()
        with pytest.raises(ToolError, match="Lottery service unavailable"):
            lottery_grpc.analyze(form=[1, 2, 3, 4, 5, 6])

    def test_mock_not_configured_raises(self, monkeypatch):
        # Set a mock client with no generate_form function.
        lottery_grpc.set_mock_client({})
        lottery_grpc.invalidate_tool_cache()
        try:
            with pytest.raises(ToolError, match="Mock generate_form not configured"):
                lottery_grpc.generate_form(how_many=1, form_type=6)
        finally:
            lottery_grpc.reset_mock_client()
