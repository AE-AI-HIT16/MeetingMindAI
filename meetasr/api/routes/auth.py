"""API routes xác thực (Authentication) — Phase 2.

Flow hoạt động:
    1. Frontend (next-auth) lấy ``id_token`` từ Google sau khi user đăng nhập.
    2. Frontend gọi ``POST /v1/auth/google`` với ``id_token`` trong body.
    3. Backend xác minh ``id_token`` với Google (gọi tokeninfo endpoint).
    4. Backend tạo User mới (hoặc cập nhật last_login_at nếu đã tồn tại).
    5. Backend ký JWT nội bộ và trả về cho frontend.
    6. Frontend lưu JWT, đính kèm vào mọi request sau: ``Authorization: Bearer <jwt>``.

Endpoints:
    POST /v1/auth/google  — đổi Google id_token lấy JWT nội bộ
    GET  /v1/auth/me      — trả về thông tin User đang đăng nhập
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Annotated

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session, select

from meetasr.api.auth_deps import get_current_user
from meetasr.db.connection import get_db
from meetasr.db.user_model import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Cấu hình JWT nội bộ
# ---------------------------------------------------------------------------

_JWT_SECRET: str = os.environ.get("JWT_SECRET", "change-me-in-production")
_JWT_ALGORITHM: str = "HS256"
_JWT_EXPIRE_DAYS: int = int(os.environ.get("JWT_EXPIRE_DAYS", "30"))

# URL Google dùng để xác minh id_token (không cần client secret)
_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SyncUserRequest(BaseModel):
    """Body của POST /v1/auth/sync."""
    provider: str
    provider_id: str
    email: str
    name: str
    avatar_url: str | None = None
    sync_secret: str  # Dùng để xác thực request từ Next.js


class AuthResponse(BaseModel):
    """Response trả về sau khi xác thực thành công."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int          # giây
    user: "UserInfo"


class UserInfo(BaseModel):
    """Thông tin user trả về kèm token."""
    id: str
    email: str
    name: str
    avatar_url: str | None


# ---------------------------------------------------------------------------
# Helper: ký JWT nội bộ
# ---------------------------------------------------------------------------

def _create_jwt(user: User) -> tuple[str, int]:
    """Ký JWT nội bộ cho user.

    Args:
        user: User đã xác thực.

    Returns:
        Tuple (token_string, expires_in_seconds).
    """
    expire_seconds = _JWT_EXPIRE_DAYS * 24 * 3600
    payload = {
        "sub": user.id,
        "email": user.email,
        "name": user.name,
        "exp": datetime.utcnow() + timedelta(seconds=expire_seconds),
        "iat": datetime.utcnow(),
    }
    token = jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)
    return token, expire_seconds


# ---------------------------------------------------------------------------
# POST /v1/auth/sync — đồng bộ user từ NextAuth lấy JWT nội bộ
# ---------------------------------------------------------------------------

@router.post("/sync", response_model=AuthResponse, status_code=status.HTTP_200_OK)
async def sync_user(
    body: SyncUserRequest,
    db: Annotated[Session, Depends(get_db)],
) -> AuthResponse:
    """Đồng bộ User từ NextAuth và cấp JWT nội bộ.

    Endpoint này chỉ được gọi từ server-side của Next.js (NextAuth callback).
    Xác thực thông qua `sync_secret` khớp với `JWT_SECRET`.
    """
    if body.sync_secret != _JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid sync secret.",
        )

    # Tìm User trong DB
    user = db.exec(
        select(User)
        .where(User.provider == body.provider)
        .where(User.provider_id == body.provider_id)
    ).first()

    if user is None:
        user = User(
            provider=body.provider,
            provider_id=body.provider_id,
            email=body.email,
            name=body.name,
            avatar_url=body.avatar_url,
        )
        db.add(user)
        logger.info("Tao User moi: provider=%s email=%s", body.provider, body.email)
    else:
        # Cập nhật thông tin có thể thay đổi
        user.name = body.name
        user.email = body.email
        user.avatar_url = body.avatar_url
        user.last_login_at = datetime.utcnow()
        logger.info("User dang nhap lai: provider=%s email=%s", body.provider, body.email)

    db.commit()
    db.refresh(user)

    # Ký JWT nội bộ
    token, expires_in = _create_jwt(user)

    return AuthResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserInfo(
            id=user.id,
            email=user.email,
            name=user.name,
            avatar_url=user.avatar_url,
        ),
    )



# ---------------------------------------------------------------------------
# GET /v1/auth/me — thông tin user đang đăng nhập
# ---------------------------------------------------------------------------

@router.get("/me", response_model=UserInfo)
def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserInfo:
    """Trả về thông tin của user đang đăng nhập.

    Yêu cầu header: ``Authorization: Bearer <access_token>``

    Args:
        current_user: User được xác thực từ JWT (Dependency).

    Returns:
        Thông tin user hiện tại.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bạn chưa đăng nhập.",
        )
    return UserInfo(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        avatar_url=current_user.avatar_url,
    )


# ---------------------------------------------------------------------------
# POST /v1/auth/refresh — gia hạn JWT nội bộ (duy trì đăng nhập)
# ---------------------------------------------------------------------------

@router.post("/refresh", response_model=AuthResponse, status_code=status.HTTP_200_OK)
def refresh_token(
    current_user: Annotated[User, Depends(get_current_user)],
) -> AuthResponse:
    """Gia hạn JWT nội bộ mà không cần đăng nhập lại OAuth.

    Client gọi endpoint này trước khi JWT sắp hết hạn để nhận token mới.
    Yêu cầu header: ``Authorization: Bearer <access_token>``

    Returns:
        ``AuthResponse`` gồm ``access_token`` mới và thông tin user.

    Raises:
        HTTPException 401: Nếu không có token hoặc token hết hạn / không hợp lệ.
    """
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bạn chưa đăng nhập.",
        )

    # Cập nhật last_login_at mỗi lần refresh thành công
    current_user.last_login_at = datetime.utcnow()

    # Ký JWT mới với thời hạn đầy đủ
    token, expires_in = _create_jwt(current_user)

    logger.info("Refresh token thành công: user_id=%s", current_user.id)

    return AuthResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserInfo(
            id=current_user.id,
            email=current_user.email,
            name=current_user.name,
            avatar_url=current_user.avatar_url,
        ),
    )


# ---------------------------------------------------------------------------
# POST /v1/auth/logout — ghi nhận đăng xuất
# ---------------------------------------------------------------------------

class LogoutResponse(BaseModel):
    """Response sau khi đăng xuất."""
    message: str


@router.post("/logout", response_model=LogoutResponse, status_code=status.HTTP_200_OK)
def logout(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> LogoutResponse:
    """Ghi nhận đăng xuất và cập nhật thời gian hoạt động cuối.

    Frontend gọi endpoint này trước khi xóa session cookie của NextAuth.
    Yêu cầu header: ``Authorization: Bearer <access_token>``

    Note:
        JWT là stateless — token cũ vẫn hợp lệ cho đến khi hết hạn tự nhiên.
        Lớp bảo mật thực sự là NextAuth xóa session cookie phía client.

    Returns:
        Thông báo đăng xuất thành công.
    """
    if current_user is None:
        # Guest logout — không cần làm gì
        return LogoutResponse(message="Đã đăng xuất.")

    logger.info(
        "User đăng xuất: user_id=%s email=%s provider=%s",
        current_user.id,
        current_user.email,
        current_user.provider,
    )

    return LogoutResponse(message=f"Tạm biệt, {current_user.name}!")
