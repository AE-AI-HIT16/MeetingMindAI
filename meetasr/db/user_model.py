"""Model User dành cho xác thực qua Google OAuth.

Mỗi User tương ứng với một tài khoản Google.
``google_id`` (sub claim trong id_token) là định danh bất biến,
không thay đổi kể cả khi người dùng đổi email.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def _new_uuid() -> str:
    return str(uuid.uuid4())


class User(SQLModel, table=True):
    """Lưu trữ thông tin người dùng đã xác thực qua OAuth."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("provider", "provider_id", name="uq_provider_id"),
        {"extend_existing": True},
    )

    id: str = Field(default_factory=_new_uuid, primary_key=True)

    # Xác thực OAuth
    provider: str = Field(index=True)         # "google", "github"
    provider_id: str = Field(index=True)      # ID trả về từ provider
    email: str = Field(index=True)
    name: str
    avatar_url: Optional[str] = None          # Ảnh đại diện

    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: datetime = Field(default_factory=datetime.utcnow)
