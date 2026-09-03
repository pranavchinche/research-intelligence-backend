"""Authentication utilities.

Per the architecture (Section 24), auth uses Supabase/Firebase JWT.
For the MVP/demo scope, authentication is optional — the system
works without it. When a JWT is present, it is validated; when
absent, requests proceed as anonymous.

This module provides the middleware and dependency injection
needed to support auth when configured.
"""

import logging

from fastapi import Request, HTTPException


logger = logging.getLogger(__name__)


async def get_current_user_id(request: Request) -> str | None:
    """Extract user ID from JWT if present.

    Returns None for unauthenticated requests (anonymous mode).
    """
    auth_header = request.headers.get("authorization", "")

    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:].strip()

    if not token:
        return None

    # TODO: Validate JWT against Supabase/Firebase when auth
    # is configured. For now, treat all tokens as anonymous.
    logger.debug(
        "Auth token present but JWT validation not configured"
    )
    return None


async def require_auth(request: Request) -> str:
    """Require a valid JWT. Raises 401 if not present/valid.

    Use this dependency on endpoints that MUST have a user.
    """
    user_id = await get_current_user_id(request)
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )
    return user_id
