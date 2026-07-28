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

class GoogleAuthRequest(BaseModel):
    """Body của POST /v1/auth/google."""
    id_token: str   # id_token lấy từ next-auth session.id_token


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
# POST /v1/auth/google — đổi Google id_token lấy JWT nội bộ
# ---------------------------------------------------------------------------

@router.post("/google", response_model=AuthResponse, status_code=status.HTTP_200_OK)
async def login_with_google(
    body: GoogleAuthRequest,
    db: Annotated[Session, Depends(get_db)],
) -> AuthResponse:
    """Xác thực người dùng bằng Google id_token và cấp JWT nội bộ.

    Frontend gọi endpoint này ngay sau khi next-auth trả về session.
    id_token là JWT do Google ký, chứa thông tin người dùng.

    Args:
        body: Body chứa ``id_token`` từ Google.
        db:   DB session.

    Returns:
        ``AuthResponse`` gồm ``access_token`` (JWT nội bộ) và thông tin user.

    Raises:
        HTTPException 401: id_token không hợp lệ hoặc đã hết hạn.
        HTTPException 502: Không thể kết nối tới Google để xác minh token.
    """
    # Bước 1: Xác minh id_token với Google
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                _GOOGLE_TOKENINFO_URL,
                params={"id_token": body.id_token},
            )
    except httpx.RequestError as exc:
        logger.error("Khong the ket noi toi Google tokeninfo: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Khong the xac minh token voi Google. Thu lai sau.",
        ) from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google id_token khong hop le hoac da het han.",
        )

    google_data = resp.json()

    # Bước 2: Trích xuất thông tin từ payload Google
    google_id: str = google_data.get("sub", "")
    email: str = google_data.get("email", "")
    name: str = google_data.get("name", email.split("@")[0])
    avatar_url: str | None = google_data.get("picture")

    if not google_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token thieu thong tin can thiet (sub hoac email).",
        )

    # Bước 3: Tìm User trong DB hoặc tạo mới
    user = db.exec(select(User).where(User.google_id == google_id)).first()

    if user is None:
        user = User(
            google_id=google_id,
            email=email,
            name=name,
            avatar_url=avatar_url,
        )
        db.add(user)
        logger.info("Tao User moi: email=%s google_id=%s", email, google_id)
    else:
        # Cập nhật thông tin có thể thay đổi (tên, avatar)
        user.name = name
        user.avatar_url = avatar_url
        user.last_login_at = datetime.utcnow()
        logger.info("User dang nhap lai: email=%s", email)

    db.commit()
    db.refresh(user)

    # Bước 4: Ký JWT nội bộ và trả về
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
    return UserInfo(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        avatar_url=current_user.avatar_url,
    )
