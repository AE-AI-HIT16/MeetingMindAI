"""Lớp cơ sở trừu tượng (Abstract base class) cho tất cả các backend lưu trữ.

Mọi backend cụ thể (local filesystem, S3/MinIO, v.v.) đều phải triển khai interface này
để phần còn lại của ứng dụng không bị phụ thuộc vào hệ thống lưu trữ bên dưới
(storage-agnostic).

Các nguyên tắc thiết kế:
- ``save`` / ``load`` / ``delete`` là các phương thức bắt buộc duy nhất.
- ``public_url`` trả về một URL mà frontend có thể sử dụng trực tiếp (presigned URL
  đối với S3, hoặc đường dẫn API proxy đối với local backend).
- Tất cả các phương thức đều là ``async`` để tránh làm nghẽn event loop của
  FastAPI khi thực hiện các tác vụ I/O
- An toàn bộ nhớ: ``save`` chấp nhận đầu vào là một file stream để hỗ trợ các file
  lớn (>1GB) mà không gây ra lỗi tràn RAM (Out-Of-Memory - OOM).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import BinaryIO, Union


class StorageBackend(ABC):
    """Interface mà mọi hệ thống lưu trữ (storage backend) phải triển khai.

    Nơi gọi (callers) tuyệt đối không nên phụ thuộc trực tiếp vào một class backend
    cụ thể; hãy sử dụng :func:`meetasr.storage.get_storage` để lấy instance.
    """

    @abstractmethod
    async def save(self, file_data: Union[BinaryIO, bytes], filename: str) -> str:
        """Lưu trữ dữ liệu dạng raw bytes hoặc file stream và trả về một khóa lưu trữ.

        Để ngăn chặn lỗi tràn RAM (OOM) khi xử lý các file media lớn (ví dụ: file
        video 1GB), nơi gọi nên truyền vào một luồng dữ liệu (file-like stream - BinaryIO)
        thay vì đọc toàn bộ file vào bộ nhớ dưới dạng bytes.
        Bản thân backend khi triển khai phải xử lý việc đọc luồng dữ liệu này theo
        từng phần nhỏ (chunks) (ví dụ: thông qua S3 Multipart Upload).

        Khóa (key) trả về là một định danh ẩn (opaque identifier) có thể được truyền
        ngược lại cho các phương thức :meth:`load`, :meth:`delete`, và :meth:`public_url`.

        Args:
            file_data: Dữ liệu dạng byte thô (dành cho các file text/docs nhỏ) hoặc
                       một file-like stream (BinaryIO) dành cho các file media lớn.
            filename:  Tên file gốc, được dùng để trích xuất khóa lưu trữ
                       (ví dụ: ``"meeting.mp4"``).

        Returns:
            Khóa lưu trữ (ví dụ: đường dẫn tương đối hoặc S3 object key).

        Raises:
            OSError: Nếu ghi dữ liệu vào đĩa cục bộ thất bại.
            RuntimeError: Nếu thao tác tải lên S3/MinIO thất bại.
        """

    @abstractmethod
    async def load(self, key: str) -> bytes:
        """Truy xuất dữ liệu bytes thô của một đối tượng đã được lưu trữ trước đó.

        Lưu ý: Phương thức này tải TOÀN BỘ đối tượng vào bộ nhớ RAM. Nó được thiết kế
        chủ yếu dành cho các file nhỏ (như tài liệu Markdown hoặc JSON metadata).
        Để phục vụ các file media lớn (video/audio) cho frontend, hãy sử dụng
        :meth:`public_url` để cho phép trình duyệt tải trực tiếp (streaming).

        Args:
            key: Khóa lưu trữ được trả về từ phương thức :meth:`save`.

        Returns:
            Dữ liệu bytes thô của file.

        Raises:
            FileNotFoundError: Nếu khóa không tồn tại trong backend.
        """

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Xóa vĩnh viễn một đối tượng đã lưu trữ.

        Việc xóa một khóa không tồn tại sẽ không sinh ra lỗi (tính idempotent).
        Đối với S3/MinIO, thao tác này sẽ kích hoạt việc xóa cứng (hard delete)
        đối tượng, ngăn chặn triệt để tình trạng rò rỉ rác lưu trữ (storage leakage).

        Args:
            key: Khóa lưu trữ được trả về từ phương thức :meth:`save`.
        """

    @abstractmethod
    def public_url(self, key: str) -> str:
        """Trả về một URL mà frontend có thể sử dụng để truy cập trực tiếp vào file.

        Đối với local backend, đây là một đường dẫn API proxy
        (ví dụ: ``/v1/sources/{id}/media``).
        Đối với S3/MinIO, phương thức này sinh ra một presigned URL bảo mật với thời
        gian sống (TTL) có thể cấu hình, cho phép trình duyệt stream video một cách
        mượt mà và an toàn.

        Args:
            key: Khóa lưu trữ được trả về từ phương thức :meth:`save`.

        Returns:
            Một URL đầy đủ (fully-qualified) hoặc đường dẫn API tuyệt đối.
        """

    def needs_redirect(self) -> bool:
        """Cho biết backend hỗ trợ presigned URL redirect thay vì stream bytes.

        Trả về ``True`` nghĩa là endpoint ``/media`` nên dùng HTTP 307 redirect
        sang :meth:`public_url` (presigned URL của MinIO/S3), thay vì đọc toàn
        bộ file vào RAM rồi stream lại qua FastAPI.

        Mặc định ``False`` (LocalStorage stream bytes qua API).
        Override trong ``S3Storage`` để trả ``True``.

        Returns:
            ``True`` nếu backend hỗ trợ presigned redirect; ``False`` nếu cần stream.
        """
        return False