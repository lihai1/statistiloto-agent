"""Unit tests for the ToolResolver."""
import pytest

from app.normalizer import NormalizedRequest
from app.tool_resolver import (
    resolve,
    OPERATION_SPECS,
    _normalize_strength,
    _validate_group_size,
    _validate_numbers,
    _validate_sql,
)


class TestNormalizeStrength:
    def test_hot_string(self):
        assert _normalize_strength("hot") == 2

    def test_cold_string(self):
        assert _normalize_strength("cold") == 1

    def test_integer_2(self):
        assert _normalize_strength(2) == 2

    def test_integer_1(self):
        assert _normalize_strength(1) == 1

    def test_invalid_none(self):
        assert _normalize_strength("warm") is None

    def test_string_digit(self):
        assert _normalize_strength("2") == 2


class TestValidation:
    def test_group_size_valid(self):
        ok, _ = _validate_group_size(2)
        assert ok is True

    def test_group_size_too_large(self):
        ok, err = _validate_group_size(7)
        assert ok is False
        assert "out of range" in err

    def test_group_size_string(self):
        ok, _ = _validate_group_size("3")
        assert ok is True

    def test_numbers_valid(self):
        ok, _ = _validate_numbers([7, 11, 17, 24, 31, 36])
        assert ok is True

    def test_numbers_empty(self):
        ok, err = _validate_numbers([])
        assert ok is False

    def test_sql_select(self):
        ok, _ = _validate_sql("SELECT * FROM users")
        assert ok is True

    def test_sql_insert_blocked(self):
        ok, err = _validate_sql("INSERT INTO users VALUES (1)")
        assert ok is False

    def test_sql_update_blocked(self):
        ok, err = _validate_sql("UPDATE users SET x=1")
        assert ok is False


class TestResolveStatistics:
    def test_hot_pairs_ready(self):
        req = NormalizedRequest(
            request_kind="statistics",
            operation="group_frequency",
            group_size=2,
            strength="hot",
            how_many=10,
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.operation == "group_frequency"
        assert res.tool == "get_statistics"
        assert res.execution_ready is True
        assert res.args["group_size"] == 2
        assert res.args["strength"] == 2
        assert res.args["how_many"] == 10

    def test_hot_missing_group_size_not_ready(self):
        req = NormalizedRequest(
            request_kind="statistics",
            operation="group_frequency",
            strength="hot",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is False
        assert res.missing_hint is not None
        assert "group size" in res.missing_hint.lower() or "גודל קבוצה" in res.missing_hint

    def test_invalid_group_size(self):
        req = NormalizedRequest(
            request_kind="statistics",
            operation="group_frequency",
            group_size=7,
            strength="hot",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is False
        assert any("out of range" in e for e in res.validation_errors)

    def test_cold_triples_he_ready(self):
        req = NormalizedRequest(
            request_kind="statistics",
            operation="group_frequency",
            group_size=3,
            strength="cold",
            how_many=10,
            language="he",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is True
        assert res.args["strength"] == 1


class TestResolveAnalyze:
    def test_analyze_ready(self):
        req = NormalizedRequest(
            request_kind="number_analysis",
            operation="analyze_numbers",
            numbers=[7, 11, 17, 24, 31, 36],
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.tool == "analyze"
        assert res.execution_ready is True
        assert res.args["form"] == [7, 11, 17, 24, 31, 36]

    def test_analyze_missing_numbers_not_ready(self):
        req = NormalizedRequest(
            request_kind="number_analysis",
            operation="analyze_numbers",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is False


class TestResolveGenerateForm:
    def test_generate_ready(self):
        req = NormalizedRequest(
            request_kind="form_generation",
            operation="generate_form",
            how_many=3,
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.tool == "generate_form"
        assert res.execution_ready is True
        assert res.args["form_type"] == 6
        assert res.args["strength"] == 2

    def test_generate_missing_how_many_not_ready(self):
        req = NormalizedRequest(
            request_kind="form_generation",
            operation="generate_form",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is False


class TestResolveEditFile:
    def test_edit_file_with_content(self):
        req = NormalizedRequest(
            request_kind="admin_operation",
            operation="edit_file",
            file_path="app/main.py",
            content="print('hello')",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.tool == "edit_file"
        assert res.execution_ready is True

    def test_edit_file_with_old_new(self):
        req = NormalizedRequest(
            request_kind="admin_operation",
            operation="edit_file",
            file_path="app/main.py",
            old_string="old",
            new_string="new",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is True

    def test_edit_file_missing_content(self):
        req = NormalizedRequest(
            request_kind="admin_operation",
            operation="edit_file",
            file_path="app/main.py",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is False
        assert res.missing_hint is not None


class TestResolveAdmin:
    def test_query_db_missing_sql(self):
        req = NormalizedRequest(
            request_kind="admin_operation",
            operation="query_db",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is False

    def test_query_db_with_sql(self):
        req = NormalizedRequest(
            request_kind="admin_operation",
            operation="query_db",
            sql="SELECT * FROM agent.token_usage LIMIT 10",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is True

    def test_read_code_ready(self):
        req = NormalizedRequest(
            request_kind="admin_operation",
            operation="read_code",
            file_path="app/main.py",
            language="en",
            confidence=0.9,
        )
        res = resolve(req)
        assert res.execution_ready is True


class TestOutOfScope:
    def test_out_of_scope(self):
        req = NormalizedRequest(
            request_kind="out_of_scope",
            operation=None,
            language="en",
            confidence=1.0,
        )
        res = resolve(req)
        assert res.tool is None
        assert res.execution_ready is False

    def test_ambiguous(self):
        req = NormalizedRequest(
            request_kind="ambiguous",
            operation=None,
            language="en",
            confidence=0.4,
        )
        res = resolve(req)
        assert res.tool is None
