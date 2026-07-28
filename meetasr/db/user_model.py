"""Model User dành cho xác thực qua Google OAuth.

Mỗi User tương ứng với một tài khoản Google.
``google_id`` (sub claim trong id_token) là định danh bất biến,
không thay đổi kể cả khi người dùng đổi email.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


def _new_uuid() -> str:
    return str(uuid.uuid4())


class User(SQLModel, table=True):
    """Lưu trữ thông tin người dùng đã xác thực qua Google OAuth."""

    __tablename__ = "users"

    id: str = Field(default_factory=_new_uuid, primary_key=True)

    # Thông tin từ Google id_token
    google_id: str = Field(unique=True, index=True)   # "sub" claim — bất biến
    email: str = Field(index=True)
    name: str
    avatar_url: Optional[str] = None                  # "picture" claim

    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: datetime = Field(default_factory=datetime.utcnow)
