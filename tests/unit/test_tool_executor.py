"""Unit tests for ToolExecutor."""
import os
import pytest

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.security import create_test_jwt, _extract_tier
from app.tool_executor import (
    execute_tool,
    _reconstruct_claims,
)


class TestExtractTier:
    def test_tier_from_payload(self):
        assert _extract_tier({"tier": "admin"}) == "admin"
        assert _extract_tier({"groups": ["/paid"]}) == "paid"
        assert _extract_tier({"realm_access": {"roles": ["admin"]}}) == "admin"
        assert _extract_tier({}) == "free"


class TestReconstructClaims:
    def test_free_claims(self):
        token = create_test_jwt(sub="free-user", tier="free")
        claims = _reconstruct_claims(token)
        assert claims.tier == "free"
        assert claims.sub == "free-user"

    def test_admin_claims(self):
        token = create_test_jwt(sub="admin-user", tier="admin")
        claims = _reconstruct_claims(token)
        assert claims.tier == "admin"


class TestExecuteToolAuthorization:
    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        from app.tools import lottery_grpc, saved_numbers
        lottery_grpc.reset_mock_client()
        saved_numbers.reset_mock_client()
        yield
        lottery_grpc.reset_mock_client()
        saved_numbers.reset_mock_client()

    def test_free_can_execute_allowed_tool(self):
        from app.tools import lottery_grpc
        lottery_grpc.set_mock_client({
            "get_statistics": lambda **kw: {"groups": []},
        })
        token = create_test_jwt(sub="free-user", tier="free")
        result = execute_tool("get_statistics", {"how_many": 10, "group_size": 2, "strength": 2}, token)
        assert "error" not in result or "groups" in result

    def test_free_cannot_execute_save_numbers(self):
        token = create_test_jwt(sub="free-user", tier="free")
        with pytest.raises(PermissionError):
            execute_tool("save_numbers", {"numbers": [1, 2, 3]}, token)

    def test_paid_can_execute_save_numbers(self):
        from app.tools import saved_numbers
        saved_numbers.set_mock_client({
            "save_numbers": lambda **kw: {"saved": True},
        })
        token = create_test_jwt(sub="paid-user", tier="paid")
        result = execute_tool("save_numbers", {"numbers": [1, 2, 3]}, token)
        assert result.get("saved") is True

    def test_unknown_tool(self):
        token = create_test_jwt(sub="admin-user", tier="admin")
        with pytest.raises(PermissionError):
            execute_tool("nonexistent_tool", {}, token)

    def test_paid_can_execute_simulate(self):
        from app.tools import lottery_grpc
        lottery_grpc.set_mock_client({
            "simulate": lambda **kw: {"draws": [], "summary": {"total_draws": 0}},
        })
        token = create_test_jwt(sub="paid-user", tier="paid")
        result = execute_tool("simulate", {"form": [1, 2, 3, 4, 5, 6]}, token)
        assert "summary" in result

    def test_free_cannot_execute_simulate(self):
        token = create_test_jwt(sub="free-user", tier="free")
        with pytest.raises(PermissionError):
            execute_tool("simulate", {"form": [1, 2, 3, 4, 5, 6]}, token)
