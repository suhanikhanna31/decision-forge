"""
Shared-secret JWT verification for tokens issued by the Django auth-service
(djangorestframework-simplejwt, HS256). This lets ml-service authenticate
requests without a network call back to Django on every decision.
"""
import os
from typing import Optional

import jwt
from fastapi import Header, HTTPException

DJANGO_SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-insecure-secret-key-change-me")


class CurrentUser:
    def __init__(self, user_id: str, username: str, role: str, organization: str):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.organization = organization


def get_current_user(authorization: Optional[str] = Header(default=None)) -> CurrentUser:
    """FastAPI dependency: require a valid Bearer JWT minted by auth-service."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <token> header")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, DJANGO_SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}")

    if payload.get("token_type") != "access":
        raise HTTPException(status_code=401, detail="Refresh tokens cannot be used to authenticate requests")

    return CurrentUser(
        user_id=str(payload.get("user_id")),
        username=payload.get("username", ""),
        role=payload.get("role", "viewer"),
        organization=payload.get("organization", ""),
    )
