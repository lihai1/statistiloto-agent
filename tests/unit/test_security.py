"""Unit tests for JWT security — no DB needed."""

import os
import pytest

os.environ["JWT_VERIFY"] = "false"
os.environ["LLM_MOCK"] = "true"

from app.security import validate_jwt, create_test_jwt, JWTError, _extract_tier


class TestCreateTestJWT:
    def test_creates_valid_jwt(self):
        token = create_test_jwt(sub="user-123", tier="free")
        assert token.count(".") == 2  # JWT has 3 parts

    def test_admin_token_has_admin_tier(self):
        token = create_test_jwt(sub="admin-1", tier="admin")
        import jwt
        claims = jwt.decode(token, "test-secret", algorithms=["HS256"],
                            options={"verify_aud": False})
        assert claims["tier"] == "admin"
        assert "/admins" in claims["groups"]


class TestValidateJWT:
    def test_valid_free_token(self):
        token = create_test_jwt(sub="user-1", tier="free")
        claims = validate_jwt(f"Bearer {token}")
        assert claims.sub == "user-1"
        assert claims.tier == "free"

    def test_valid_paid_token(self):
        token = create_test_jwt(sub="user-2", tier="paid")
        claims = validate_jwt(f"Bearer {token}")
        assert claims.tier == "paid"

    def test_valid_admin_token(self):
        token = create_test_jwt(sub="admin-1", tier="admin")
        claims = validate_jwt(f"Bearer {token}")
        assert claims.tier == "admin"

    def test_missing_header(self):
        with pytest.raises(JWTError, match="Missing"):
            validate_jwt("")

    def test_wrong_scheme(self):
        with pytest.raises(JWTError, match="Bearer"):
            validate_jwt("Basic abc123")

    def test_malformed_token(self):
        with pytest.raises(JWTError):
            validate_jwt("Bearer not-a-jwt")


class TestExtractTier:
    def test_explicit_tier_claim(self):
        assert _extract_tier({"tier": "admin"}) == "admin"

    def test_groups_admin(self):
        assert _extract_tier({"groups": ["/admins"]}) == "admin"

    def test_groups_paid(self):
        assert _extract_tier({"groups": ["/paid"]}) == "paid"

    def test_groups_users(self):
        assert _extract_tier({"groups": ["/users"]}) == "free"

    def test_roles_admin(self):
        assert _extract_tier({"realm_access": {"roles": ["ADMIN"]}}) == "admin"

    def test_default_free(self):
        assert _extract_tier({}) == "free"
