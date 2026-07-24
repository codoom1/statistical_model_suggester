"""Persistent document storage for local development and Vercel."""

import mimetypes
import os
from pathlib import Path
from uuid import uuid4

from flask import current_app
from werkzeug.utils import secure_filename


MAX_UPLOAD_BYTES = 4 * 1024 * 1024
LOCAL_PREFIX = "local:"


class StorageConfigurationError(RuntimeError):
    """Raised when production object storage is not configured."""


def _read_limited(file_storage) -> bytes:
    data = file_storage.stream.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Resume files must be smaller than 4 MB.")
    if not data:
        raise ValueError("The uploaded resume is empty.")
    return data


def store_resume(file_storage, user_id: int) -> str:
    """Store an uploaded resume and return its durable reference."""
    filename = secure_filename(file_storage.filename or "resume")
    extension = Path(filename).suffix.lower()
    object_name = f"expert-resumes/{user_id}/{uuid4().hex}{extension}"
    data = _read_limited(file_storage)
    content_type = (
        file_storage.mimetype
        or mimetypes.guess_type(filename)[0]
        or "application/octet-stream"
    )

    if os.environ.get("BLOB_READ_WRITE_TOKEN"):
        from vercel.blob import BlobClient

        with BlobClient() as client:
            result = client.put(
                object_name,
                data,
                access="private",
                content_type=content_type,
                add_random_suffix=False,
            )
        return result.url

    if os.environ.get("VERCEL") or os.environ.get(
        "FLASK_ENV", ""
    ).lower() == "production":
        raise StorageConfigurationError(
            "BLOB_READ_WRITE_TOKEN is required for resume uploads."
        )

    upload_root = Path(current_app.instance_path) / "uploads" / "resumes"
    upload_root.mkdir(parents=True, exist_ok=True)
    local_name = f"{user_id}-{uuid4().hex}{extension}"
    (upload_root / local_name).write_bytes(data)
    return f"{LOCAL_PREFIX}{local_name}"


def read_resume(reference: str) -> tuple[bytes, str, str]:
    """Read a stored resume as bytes, MIME type, and download filename."""
    if reference.startswith(LOCAL_PREFIX):
        filename = secure_filename(reference.removeprefix(LOCAL_PREFIX))
        if not filename:
            raise FileNotFoundError("Invalid local resume reference.")
        path = Path(current_app.instance_path) / "uploads" / "resumes" / filename
        data = path.read_bytes()
        content_type = (
            mimetypes.guess_type(filename)[0] or "application/octet-stream"
        )
        return data, content_type, filename

    if not os.environ.get("BLOB_READ_WRITE_TOKEN"):
        raise StorageConfigurationError(
            "BLOB_READ_WRITE_TOKEN is required to read this resume."
        )

    from vercel.blob import BlobClient

    with BlobClient() as client:
        result = client.get(reference, access="private")
    filename = Path(result.pathname).name or "resume"
    return (
        result.content,
        result.content_type or "application/octet-stream",
        filename,
    )
