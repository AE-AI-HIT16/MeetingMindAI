"""
Cấu hình cơ sở dữ liệu và quản lý phiên làm việc cho dự án MeetASR.

Hỗ trợ SQLite cho môi trường phát triển cục bộ (local) và PostgreSQL cho môi trường thực tế (production).
Thay đổi cấu hình thông qua biến môi trường DATABASE_URL trong tệp .env.
"""

import logging
import os
from typing import Generator

from dotenv import load_dotenv  # Bổ sung thư viện đọc file .env
from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine, select

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
    import meetasr.db.user_model  # noqa: F401  -- bang users
    SQLModel.metadata.create_all(engine)
    _ensure_unique_documents()
    logger.info("Đã khởi tạo các bảng trong cơ sở dữ liệu.")


def _ensure_unique_documents() -> None:
    """Archive duplicate documents, then enforce one row per source/mode.

    ``create_all`` does not add a newly declared constraint to an existing
    table. This small migration keeps the richest existing document, archives
    every duplicate, and creates a named unique index for existing databases.
    """
    from meetasr.db.models_phase2 import Document, DocumentDuplicateArchive

    with Session(engine) as session:
        documents = session.exec(
            select(Document).order_by(
                Document.source_id,
                Document.mode,
                Document.updated_at.desc(),
            )
        ).all()
        by_key: dict[tuple[str, str], list[Document]] = {}
        for document in documents:
            by_key.setdefault(
                (document.source_id, document.mode),
                [],
            ).append(document)

        duplicate_count = 0
        for duplicates in by_key.values():
            if len(duplicates) < 2:
                continue
            # Preserve the most useful document. A longer non-empty result is
            # preferable to a later partial result produced during rate limits.
            keep = max(
                duplicates,
                key=lambda item: (
                    len(item.markdown.strip()),
                    item.updated_at,
                ),
            )
            for duplicate in duplicates:
                if duplicate.id == keep.id:
                    continue
                session.add(
                    DocumentDuplicateArchive(
                        original_document_id=duplicate.id,
                        source_id=duplicate.source_id,
                        mode=duplicate.mode,
                        markdown=duplicate.markdown,
                        original_created_at=duplicate.created_at,
                        original_updated_at=duplicate.updated_at,
                    )
                )
                session.delete(duplicate)
                duplicate_count += 1
        session.commit()

    if duplicate_count:
        logger.warning(
            "Đã lưu trữ và loại %d Document trùng trước khi tạo unique index.",
            duplicate_count,
        )

    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "uq_documents_source_mode "
                "ON documents (source_id, mode)"
            )
        )



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
