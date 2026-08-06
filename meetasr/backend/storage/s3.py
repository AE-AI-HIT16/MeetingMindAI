"""MinIO / S3-compatible storage backend.

Uploads and downloads media files to/from a MinIO bucket using the
``boto3`` client library (S3-compatible API).

The MinIO instance is already provisioned in ``docker-compose.yml``;
make sure it is running before using this backend::

    docker-compose up -d minio_storage

Required environment variables (or pass directly to the constructor):

    MINIO_ENDPOINT          http://localhost:9000
    MINIO_ROOT_USER         admin
    MINIO_ROOT_PASSWORD     password123
    MINIO_BUCKET            meetasr-audio
    MINIO_PRESIGN_EXPIRES   3600   (optional, seconds, default 1 hour)
"""

from __future__ import annotations

import io
import logging
import uuid
from typing import BinaryIO, Optional, Union

import boto3
from botocore.exceptions import ClientError

from meetasr.backend.storage.backend import StorageBackend

logger = logging.getLogger(__name__)

# Default presigned URL expiry (seconds)
DEFAULT_PRESIGN_EXPIRES: int = 3600


class S3Storage(StorageBackend):
    """Store files on MinIO (or any S3-compatible service).

    Object key structure::

        {uuid_prefix}/{original_filename}

    e.g. ``"a1b2c3de/meeting.wav"``

    Args:
        endpoint_url:    Full URL of the MinIO endpoint, e.g. ``"http://localhost:9000"``.
        access_key:      MinIO root user / AWS access key ID.
        secret_key:      MinIO root password / AWS secret access key.
        bucket:          Target bucket name. Created automatically if absent.
        presign_expires: Presigned URL TTL in seconds (default: 3600).
        region:          AWS region name; use ``"us-east-1"`` for MinIO.
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        presign_expires: int = DEFAULT_PRESIGN_EXPIRES,
        region: str = "us-east-1",
    ) -> None:
        self._bucket = bucket
        self._presign_expires = presign_expires

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

        self._ensure_bucket()
        logger.info(
            "S3Storage initialised: endpoint=%s bucket=%s", endpoint_url, bucket
        )

    # ------------------------------------------------------------------
    # StorageBackend implementation
    # ------------------------------------------------------------------

    async def save(self, file_data: Union[BinaryIO, bytes], filename: str) -> str:
        """Upload dữ liệu lên MinIO và trả về object key.

        Hỗ trợ cả ``bytes`` (dành cho file nhỏ) lẫn ``BinaryIO`` stream
        (dành cho file video lớn — boto3 ``upload_fileobj`` tự động dùng
        S3 Multipart Upload nếu object vượt ngưỡng 8 MB).

        Args:
            file_data: Dữ liệu bytes hoặc file-like stream (ví dụ: ``UploadFile.file``).
            filename:  Tên file gốc dùng làm leaf của object key.

        Returns:
            Object key trong bucket, ví dụ: ``"a1b2c3de/meeting.mp4"``.

        Raises:
            RuntimeError: Nếu thao tác upload MinIO thất bại.
        """
        prefix = uuid.uuid4().hex[:8]
        key = f"{prefix}/{filename}"

        # Chuẩn hóa file_data thành file-like object cho upload_fileobj
        stream: BinaryIO = (
            io.BytesIO(file_data)
            if isinstance(file_data, (bytes, bytearray))
            else file_data
        )

        try:
            # upload_fileobj tự dùng Multipart Upload khi cần
            self._client.upload_fileobj(stream, self._bucket, key)
        except ClientError as exc:
            raise RuntimeError(
                f"S3Storage: failed to upload {filename!r} -> {exc}"
            ) from exc

        logger.info("S3Storage: uploaded -> s3://%s/%s", self._bucket, key)
        return key

    async def load(self, key: str) -> bytes:
        """Download and return raw bytes for a stored object.

        Args:
            key: Object key returned by :meth:`save`.

        Returns:
            Raw file bytes.

        Raises:
            FileNotFoundError: If the object does not exist in the bucket.
            RuntimeError:      If the download fails for another reason.
        """
        buf = io.BytesIO()
        try:
            self._client.download_fileobj(self._bucket, key, buf)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("NoSuchKey", "404"):
                raise FileNotFoundError(
                    f"S3Storage: key not found: {key!r}"
                ) from exc
            raise RuntimeError(
                f"S3Storage: failed to download {key!r} → {exc}"
            ) from exc

        return buf.getvalue()

    async def delete(self, key: str) -> None:
        """Delete an object from the bucket (idempotent).

        Args:
            key: Object key returned by :meth:`save`.
        """
        try:
            self._client.delete_object(Bucket=self._bucket, Key=key)
            logger.info("S3Storage: deleted s3://%s/%s", self._bucket, key)
        except ClientError as exc:
            # Deleting a non-existent object is not an error in S3 — log & continue
            logger.warning("S3Storage: delete failed for %r: %s", key, exc)

    def public_url(self, key: str, expires: Optional[int] = None) -> str:
        """Generate a presigned URL that grants temporary read access.

        The URL is valid for ``expires`` seconds (defaults to the value
        set at construction time, usually 1 hour).

        Args:
            key:     Object key returned by :meth:`save`.
            expires: Override TTL in seconds; ``None`` uses the default.

        Returns:
            Presigned HTTPS URL string.

        Raises:
            RuntimeError: If presigning fails.
        """
        ttl = expires if expires is not None else self._presign_expires
        try:
            url: str = self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=ttl,
            )
        except ClientError as exc:
            raise RuntimeError(
                f"S3Storage: failed to generate presigned URL for {key!r}: {exc}"
            ) from exc
        return url

    def needs_redirect(self) -> bool:
        """S3/MinIO ho tro presigned URL — endpoint /media nen redirect.

        Khi True, endpoint GET /sources/{id}/media se dung HTTP 307 redirect
        sang presigned URL cua MinIO thay vi doc toan bo bytes vao RAM va stream.

        Returns:
            True.
        """
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_bucket(self) -> None:
        """Create the target bucket if it does not already exist."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
            logger.debug("S3Storage: bucket %r already exists.", self._bucket)
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in ("404", "NoSuchBucket"):
                self._client.create_bucket(Bucket=self._bucket)
                logger.info("S3Storage: created bucket %r.", self._bucket)
            else:
                raise RuntimeError(
                    f"S3Storage: cannot verify bucket {self._bucket!r}: {exc}"
                ) from exc
