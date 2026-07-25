"""Storage backend abstraction for MeetASR.

Provides a unified interface for storing and retrieving media files,
regardless of the underlying storage system (local filesystem or S3/MinIO).

Usage::

    from meetasr.storage import get_storage
    from meetasr.storage.s3 import S3Storage

    storage = get_storage()                    # from env-var STORAGE_BACKEND
    key = await storage.save(b"...", "meeting.wav")
    url = storage.public_url(key)
    data = await storage.load(key)
    await storage.delete(key)
"""

from meetasr.storage.backend import StorageBackend
from meetasr.storage.local import LocalStorage
from meetasr.storage.s3 import S3Storage


def get_storage() -> StorageBackend:
    """Instantiate the active storage backend from environment variables.

    Reads ``STORAGE_BACKEND`` (default: ``"local"``) and returns the
    corresponding backend pre-configured from environment variables:

    * ``"local"``  → :class:`LocalStorage` (reads ``STORAGE_LOCAL_ROOT``)
    * ``"s3"``     → :class:`S3Storage`    (reads ``MINIO_*`` variables)

    Returns:
        A ready-to-use :class:`StorageBackend` instance.

    Raises:
        ValueError: If ``STORAGE_BACKEND`` is set to an unknown value.
    """
    import os

    backend = os.environ.get("STORAGE_BACKEND", "local").lower()

    if backend == "local":
        root = os.environ.get("STORAGE_LOCAL_ROOT", "data/media")
        return LocalStorage(root_dir=root)

    if backend == "s3":
        return S3Storage(
            endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
            access_key=os.environ.get("MINIO_ROOT_USER", "admin"),
            secret_key=os.environ.get("MINIO_ROOT_PASSWORD", "password123"),
            bucket=os.environ.get("MINIO_BUCKET", "meetasr-audio"),
        )

    raise ValueError(
        f"Unknown STORAGE_BACKEND: {backend!r}. "
        "Valid options are 'local' and 's3'."
    )


__all__ = [
    "StorageBackend",
    "LocalStorage",
    "S3Storage",
    "get_storage",
]
