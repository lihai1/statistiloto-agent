"""JWT validation against Keycloak JWKS.

In dev/test mode (JWT_VERIFY=false), accepts any well-formed JWT
and extracts claims from it without signature verification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jwt
from app.config.settings import get_settings
from fastapi import Depends, Header, HTTPException

log = logging.getLogger(__name__)


@dataclass
class TokenClaims:
    sub: str
    tier: str          # free | paid | admin
    roles: list[str]
    raw_token: str


class JWTError(Exception):
    """Raised when a JWT is invalid or missing."""
    pass


def _extract_tier(claims: dict) -> str:
    """Determine tier from JWT claims.

    Priority: explicit `tier` claim > group membership > roles > 'free'.
    """
    if "tier" in claims:
        return claims["tier"]
    groups = claims.get("groups", [])
    if "/admins" in groups:
        return "admin"
    if "/paid" in groups:
        return "paid"
    if "/users" in groups:
        return "free"
    roles = claims.get("realm_access", {}).get("roles", [])
    if "ADMIN" in roles or "admin" in roles:
        return "admin"
    if "PAID" in roles or "paid" in roles:
        return "paid"
    return "free"


def _extract_roles(claims: dict) -> list[str]:
    roles = claims.get("realm_access", {}).get("roles", [])
    if not isinstance(roles, list):
        roles = []
    return roles


def validate_jwt(authorization_header: str) -> TokenClaims:
    """Validate a Bearer token and return claims.

    Args:
        authorization_header: the raw Authorization header value
            (e.g. "Bearer eyJ...").

    Returns:
        TokenClaims with sub, tier, roles.

    Raises:
        JWTError: if the token is missing, malformed, or invalid.
    """
    if not authorization_header:
        raise JWTError("Missing Authorization header")

    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        raise JWTError("Expected 'Bearer <token>'")
    token = parts[1]

    settings = get_settings()

    if not settings.security.jwt_verify:
        # Dev/test mode — decode without verification.
        try:
            claims = jwt.decode(token, options={"verify_signature": False})
        except jwt.PyJWTError as e:
            raise JWTError(f"Malformed token: {e}") from e
    else:
        # Production — verify against Keycloak JWKS.
        try:
            # PyJWT supports JWKS via PyJWKClient.
            from jwt import PyJWKClient
            jwks_client = PyJWKClient(settings.security.jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            # The issuer in the token may differ from the internal Keycloak URL
            # (e.g. token has http://localhost/... but internal is http://auth:8080/...).
            # We verify the signature via JWKS and the audience, but skip issuer
            # verification to avoid mismatch between external and internal URLs.
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=settings.security.audience or None,
                options={"verify_iss": False},
            )
            # Issuer varies by deployment (localhost / ngrok / prod domain),
            # so strict iss verification is off — but when ALLOWED_ISSUERS is
            # configured, the token's iss must be one of the trusted values.
            allowed = settings.security.allowed_issuers
            if allowed and claims.get("iss") not in allowed:
                raise JWTError(f"Untrusted issuer: {claims.get('iss')}")
        except jwt.PyJWTError as e:
            raise JWTError(f"Invalid token: {e}") from e

    sub = claims.get("sub", "")
    if not sub:
        raise JWTError("Token missing 'sub' claim")

    return TokenClaims(
        sub=sub,
        tier=_extract_tier(claims),
        roles=_extract_roles(claims),
        raw_token=token,
    )


def require_admin(claims: TokenClaims) -> None:
    """Raise JWTError if the user is not an admin."""
    if claims.tier != "admin":
        raise JWTError(f"Admin access required, got tier={claims.tier}")


# ── FastAPI dependencies ──────────────────────────────────────
# These reusable dependencies replace the per-endpoint try/except
# boilerplate that was duplicated across all 21 endpoints in main.py.
# Using Depends() also surfaces auth requirements in the OpenAPI schema.

async def get_current_user(authorization: str = Header(...)) -> TokenClaims:
    """FastAPI dependency: validate the Bearer JWT and return TokenClaims.

    Raises HTTPException(401) for missing, malformed, or invalid tokens.
    """
    try:
        return validate_jwt(authorization)
    except JWTError as e:
        raise HTTPException(status_code=401, detail=str(e))


async def require_admin_user(claims: TokenClaims = Depends(get_current_user)) -> TokenClaims:
    """FastAPI dependency: require an admin-tier user.

    Raises HTTPException(403) only when the token is valid but the user
    is not an admin. Invalid/expired tokens are caught by
    ``get_current_user`` and return 401.
    """
    if claims.tier != "admin":
        raise HTTPException(status_code=403, detail=f"Admin access required, got tier={claims.tier}")
    return claims


# ── Test helpers ─────────────────────────────────────────────

def create_test_jwt(
    sub: str = "test-user",
    tier: str = "free",
    roles: list[str] | None = None,
    secret: str = "test-secret",
) -> str:
    """Create a signed JWT for testing (HS256, not for production).

    The tier is embedded as both a `tier` claim and in groups/roles
    so that _extract_tier works regardless of which path is used.
    """
    groups = []
    if tier == "admin":
        groups = ["/admins"]
    elif tier == "paid":
        groups = ["/paid"]
    else:
        groups = ["/users"]

    payload = {
        "sub": sub,
        "tier": tier,
        "groups": groups,
        "realm_access": {"roles": roles or [tier.upper()]},
        "iss": "test-issuer",
        "aud": "test-audience",
    }
    return jwt.encode(payload, secret, algorithm="HS256")
