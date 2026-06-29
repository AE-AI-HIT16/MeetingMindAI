"""
Cấu hình động cơ (engine) cơ sở dữ liệu và quản lý phiên làm việc (session) cho dự án MeetASR.

Hỗ trợ SQLite cho môi trường phát triển cục bộ (local) và PostgreSQL cho môi trường thực tế (production).
Thay đổi cấu hình thông qua biến môi trường DATABASE_URL:
  - SQLite (Mặc định): sqlite:///meetasr.db
  - PostgreSQL:        postgresql://user:pass@host:5432/meetasr
"""

import logging
import os
from typing import Generator

from sqlmodel import SQLModel, Session, create_engine

logger = logging.getLogger(__name__)


# Đọc DATABASE_URL từ biến môi trường; nếu không có, mặc định dùng file SQLite cục bộ.
DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///meetasr.db")

#khắc phục lỗi multi-threading của SQLite khi chạy cùng FastAPI.
_connect_args: dict = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

# Khởi tạo động cơ kết nối cơ sở dữ liệu
engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,  # Đổi thành True nếu muốn in ra log toàn bộ câu lệnh SQL (hữu ích khi debug)
)

logger.info(f"Đã khởi tạo engine cơ sở dữ liệu: {DATABASE_URL}")


def init_db() -> None:
    """Tạo tất cả các bảng dựa trên siêu dữ liệu (metadata) đã định nghĩa bằng SQLModel.

    Hàm này an toàn khi gọi nhiều lần — nó chỉ tạo ra những bảng chưa tồn tại.
    Nên được gọi một lần duy nhất khi khởi động ứng dụng (ví dụ: trong sự kiện lifespan của FastAPI).
    """
    # Import file models để SQLModel ghi nhận cấu trúc các bảng trước khi tiến hành create_all.
    from meetasr.db import models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    logger.info("Đã khởi tạo các bảng trong cơ sở dữ liệu.")



def get_db() -> Generator[Session, None, None]:
    """Cấp phát một Session của SQLModel và đảm bảo nó được đóng an toàn sau khi sử dụng xong.

    Được thiết kế chuẩn mực để dùng làm Dependency Injection trong FastAPI::

        from fastapi import Depends
        from meetasr.db.connection import get_db

        @app.get("/meetings/{id}")
        def read_meeting(id: str, db: Session = Depends(get_db)):
            return db.get(Meeting, id)
      
      
    """
    with Session(engine) as session:
        try:
            yield session
        except Exception:
            session.rollback()
            raise