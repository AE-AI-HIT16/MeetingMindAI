"""Local filesystem storage backend.

Lưu trữ các tệp media được tải lên vào một thư mục gốc (root directory)
trên ổ cứng vật lý của máy chủ. Phù hợp cho môi trường phát triển (development)
hoặc triển khai trên một máy chủ đơn lẻ.

Khóa lưu trữ (storage key) là một đường dẫn tương đối, ví dụ: ``"abc123/meeting.mp4"``.
Frontend sẽ truy xuất các tệp này thông qua API endpoint của FastAPI
``GET /v1/sources/{source_id}/media``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import BinaryIO, Union

from meetasr.storage.backend import StorageBackend

logger = logging.getLogger(__name__)

# Thư mục gốc mặc định để lưu trữ media
DEFAULT_ROOT = "data/media"


class LocalStorage(StorageBackend):
    """Lưu trữ tệp trên hệ thống tập tin cục bộ (Local Filesystem) dưới thư mục ``root_dir``.

    Cấu trúc thư mục::

        {root_dir}/
        └── {uuid_prefix}/
            └── {original_filename}

    Args:
        root_dir: Đường dẫn tuyệt đối hoặc tương đối tới thư mục gốc chứa media.
                  Thư mục này sẽ tự động được tạo nếu chưa tồn tại.
    """

    def __init__(self, root_dir: str = DEFAULT_ROOT) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("LocalStorage đã được khởi tạo tại: %s", self._root.resolve())

    # Triển khai StorageBackend

    async def save(self, file_data: Union[BinaryIO, bytes], filename: str) -> str:
        """Ghi dữ liệu ra đĩa cứng và trả về khóa lưu trữ (storage key).

        Hỗ trợ cả ``bytes`` (dành cho file nhỏ) lẫn ``BinaryIO`` stream
        (dành cho file video lớn — ghi theo từng chunk 8MB để tránh lỗi tràn RAM).
        Sử dụng ``asyncio.to_thread`` để không làm nghẽn (block) event loop của FastAPI.

        Args:
            file_data: Dữ liệu bytes hoặc file-like stream (ví dụ: ``UploadFile.file``).
            filename:  Tên file gốc dùng làm tên tệp trên đĩa.

        Returns:
            Khóa lưu trữ dạng ``"{prefix}/{filename}"``.

        Raises:
            OSError: Nếu ghi đĩa thất bại (hết dung lượng, lỗi phân quyền...).
        """
        prefix = uuid.uuid4().hex[:8]
        dest_dir = self._root / prefix
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / filename

        def _write_file() -> int:
            """Hàm đồng bộ chạy ngầm để ghi đĩa."""
            if isinstance(file_data, (bytes, bytearray)):
                dest_path.write_bytes(file_data)
                return len(file_data)

            # Đọc và ghi theo chunk 8 MB để không tràn RAM
            chunk_size = 8 * 1024 * 1024
            total_size = 0
            with dest_path.open("wb") as fout:
                while chunk := file_data.read(chunk_size):
                    fout.write(chunk)
                    total_size += len(chunk)
            return total_size

        # Chạy tác vụ ghi đĩa ở một luồng khác để không treo Server
        size = await asyncio.to_thread(_write_file)

        key = f"{prefix}/{filename}"
        logger.info("LocalStorage: Đã lưu %d bytes -> %s", size, key)
        return key

    async def load(self, key: str) -> bytes:
        """Đọc và trả về nội dung bytes thô của một tệp đã lưu trữ.

        Args:
            key: Khóa lưu trữ được trả về từ phương thức :meth:`save`.

        Returns:
            Dữ liệu bytes của tệp.

        Raises:
            FileNotFoundError: Nếu tệp không tồn tại trên ổ cứng.
        """
        path = self._root / key
        if not path.exists():
            raise FileNotFoundError(f"LocalStorage: Không tìm thấy khóa: {key!r}")

        # Đọc file trên luồng ngầm
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, key: str) -> None:
        """Xóa một tệp khỏi ổ cứng (idempotent — bỏ qua nếu tệp không tồn tại).

        Args:
            key: Khóa lưu trữ được trả về từ phương thức :meth:`save`.
        """
        path = self._root / key

        def _delete_file():
            if path.exists():
                path.unlink()
                logger.info("LocalStorage: Đã xóa %s", key)
                # Xóa luôn thư mục cha (cái chứa prefix) nếu nó đang trống
                try:
                    path.parent.rmdir()
                except OSError:
                    pass  # Thư mục không trống, bỏ qua

        await asyncio.to_thread(_delete_file)

    def public_url(self, key: str) -> str:
        """Trả về đường dẫn API proxy để frontend có thể tải/stream tệp.

        Dữ liệu thực tế sẽ được stream bởi endpoint ``GET /v1/sources/{source_id}/media``.
        Phương thức này chỉ trả về *đường dẫn tương đối* (path) để frontend tự động
        nối với domain gốc của trang web.

        Args:
            key: Khóa lưu trữ được trả về từ phương thức :meth:`save`.

        Returns:
            Đường dẫn API tuyệt đối, ví dụ: ``"/media/a1b2c3/meeting.mp4"``.
        """
        return f"/media/{key}"


    def abs_path(self, key: str) -> Path:
        """Trả về đường dẫn tuyệt đối trên hệ thống tập tin cho một khóa.

        Hữu ích khi bạn cần truyền trực tiếp đường dẫn file cho các công cụ
        bên ngoài (ví dụ: ffmpeg) thay vì đọc toàn bộ bytes vào RAM.
        """
        path = self._root / key
        if not path.exists():
            raise FileNotFoundError(f"LocalStorage: Không tìm thấy khóa: {key!r}")
        return path.resolve()