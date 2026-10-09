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
from typing import Annotated, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlmodel import Session

from meetasr.db.connection import get_db
from meetasr.db.user_model import User

logger = logging.getLogger(__name__)

# Thuật toán và secret phải trùng với auth.py khi ký JWT
_JWT_SECRET = os.environ.get("JWT_SECRET", "change-me-in-production")
_JWT_ALGORITHM = "HS256"

# auto_error=False cho phép endpoint truy cập bằng Guest (không có token)
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer)],
    db: Annotated[Session, Depends(get_db)],
) -> Optional[User]:
    """Xác minh Bearer JWT và trả về User tương ứng (nếu có).

    Dùng làm Dependency: ``current_user: User | None = Depends(get_current_user)``

    Returns:
        User đã xác thực hoặc None nếu là khách.

    Raises:
        HTTPException 401: Token có truyền nhưng hết hạn hoặc không hợp lệ.
    """
    if not credentials or not credentials.credentials:
        return None

    return resolve_token_user(credentials.credentials, db)


def resolve_token_user(token: str, db: Session) -> User:
    """Validate one internal JWT outside the HTTP Bearer dependency."""
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
    # Nếu gửi JWT nhưng user không có trong DB thì chặn luôn.
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản không tồn tại.",
        )
    return user
