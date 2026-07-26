"""
Cấu hình cơ sở dữ liệu và quản lý phiên làm việc cho dự án MeetASR.

Hỗ trợ SQLite cho môi trường phát triển cục bộ (local) và PostgreSQL cho môi trường thực tế (production).
Thay đổi cấu hình thông qua biến môi trường DATABASE_URL trong tệp .env.
"""

import logging
import os
from typing import Generator
from dotenv import load_dotenv  # Bổ sung thư viện đọc file .env

from sqlmodel import SQLModel, Session, create_engine

logger = logging.getLogger(__name__)

# Nạp biến môi trường từ tệp .env
load_dotenv()

#hệ thống sẽ ưu tiên lấy DATABASE_URL từ file .env 
# Nếu không tìm thấy file .env, nó mới lùi về dùng SQLite mặc định.
DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///meetasr.db")

# Khắc phục lỗi multi-threading của SQLite khi chạy cùng FastAPI (chỉ áp dụng nếu dùng SQLite)
_connect_args: dict = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

# Khởi tạo động cơ kết nối cơ sở dữ liệu
engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,  # Đổi thành True nếu muốn in ra log toàn bộ câu lệnh SQL (hữu ích khi debug)
)

logger.info(f"Đã khởi tạo engine cơ sở dữ liệu với URL: {DATABASE_URL}")


def init_db() -> None:
    """Tạo tất cả các bảng dựa trên siêu dữ liệu (metadata) đã định nghĩa bằng SQLModel.

    Hàm này an toàn khi gọi nhiều lần — nó chỉ tạo ra những bảng chưa tồn tại.
    Nên được gọi một lần duy nhất khi khởi động ứng dụng (ví dụ: trong sự kiện lifespan của FastAPI).
    """
    # Import cả hai file models để SQLModel ghi nhận cấu trúc TẤT CẢ các bảng
    # trước khi tiến hành create_all.
    import meetasr.db.models  # noqa: F401
    import meetasr.db.models_phase2  # noqa: F401
    SQLModel.metadata.create_all(engine)
    logger.info("Đã khởi tạo các bảng trong cơ sở dữ liệu.")



def get_db() -> Generator[Session, None, None]:
    """Cấp phát một Session của SQLModel và đảm bảo nó được đóng an toàn sau khi sử dụng xong.

    Được thiết kế chuẩn mực để dùng làm Dependency Injection trong FastAPI:

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