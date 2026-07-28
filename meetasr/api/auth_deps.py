"""Dependency FastAPI dùng để bảo vệ các endpoint cần đăng nhập.

Cách dùng trong router:
    from meetasr.api.auth_deps import get_current_user
    from meetasr.db.user_model import User

    @router.get("/me")
    def read_me(current_user: User = Depends(get_current_user)):
        return {"email": current_user.email}
"""
from __future__ import annotations

import logging
import os
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from meetasr.db.connection import get_db
from meetasr.db.user_model import User
from sqlmodel import Session, select

logger = logging.getLogger(__name__)

# Thuật toán và secret phải trùng với auth.py khi ký JWT
_JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production")
_JWT_ALGORITHM = "HS256"

_bearer = HTTPBearer(auto_error=True)


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Xác minh Bearer JWT và trả về User tương ứng.

    Dùng làm Dependency trong các endpoint cần đăng nhập:
        ``current_user: User = Depends(get_current_user)``

    Args:
        credentials: Header ``Authorization: Bearer <token>``.
        db:          DB session.

    Returns:
        User đã xác thực.

    Raises:
        HTTPException 401: Token thiếu, hết hạn hoặc không hợp lệ.
        HTTPException 401: User không tồn tại trong DB (đã bị xóa).
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
        user_id: str = payload.get("sub", "")
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token đã hết hạn. Vui lòng đăng nhập lại.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ.",
        )

    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản không tồn tại.",
        )
    return user
