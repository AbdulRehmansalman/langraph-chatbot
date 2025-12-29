"""
FastAPI Dependencies
====================
Shared dependency injection for API routes.

This module provides reusable dependencies for:
- Authentication (token validation, user retrieval)
- Authorization (permission checks)
"""

import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.security import verify_token
from app.services.supabase_client import supabase_client
from app.models.schemas import User

logger = logging.getLogger(__name__)
security = HTTPBearer()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Validate JWT token and return user ID.

    This is a lightweight dependency for routes that only need the user ID.
    Use get_current_user() if you need the full user object.

    Args:
        credentials: HTTP Bearer token from request header

    Returns:
        User ID string

    Raises:
        HTTPException: 401 if token is invalid or missing
    """
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID"
            )

        return user_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Validate JWT token and return full user object from database.

    Use this when you need user details beyond just the ID.
    For performance-critical routes, prefer get_current_user_id().

    Args:
        credentials: HTTP Bearer token from request header

    Returns:
        User object with full profile data

    Raises:
        HTTPException: 401 if token invalid, 404 if user not found
    """
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )

        # Get user from database
        db_user = supabase_client.table("users").select("*").eq("id", user_id).execute()

        if not db_user.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user_data = db_user.data[0]
        return User(
            id=user_data["id"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            created_at=user_data["created_at"],
            is_active=user_data["is_active"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )