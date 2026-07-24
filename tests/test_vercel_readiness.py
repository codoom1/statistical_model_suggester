from io import BytesIO

import pytest
from flask import url_for
from werkzeug.datastructures import FileStorage

from utils.storage import MAX_UPLOAD_BYTES, read_resume, store_resume


def test_vercel_uses_cdn_static_endpoint(monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://user:password@localhost/database"
    )
    monkeypatch.setenv("SECRET_KEY", "production-test-secret")

    from app import create_app

    vercel_app = create_app()
    assert vercel_app.static_folder is None
    with vercel_app.test_request_context():
        assert url_for("static", filename="styles.css") == "/static/styles.css"


def test_local_resume_storage_round_trip(app, monkeypatch, tmp_path):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("BLOB_READ_WRITE_TOKEN", raising=False)
    monkeypatch.setenv("FLASK_ENV", "development")
    app.instance_path = str(tmp_path)
    upload = FileStorage(
        stream=BytesIO(b"%PDF-test"),
        filename="resume.pdf",
        content_type="application/pdf",
    )

    with app.app_context():
        reference = store_resume(upload, user_id=42)
        data, content_type, filename = read_resume(reference)

    assert reference.startswith("local:")
    assert data == b"%PDF-test"
    assert content_type == "application/pdf"
    assert filename.endswith(".pdf")


def test_resume_upload_size_limit(app, monkeypatch):
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("BLOB_READ_WRITE_TOKEN", raising=False)
    upload = FileStorage(
        stream=BytesIO(b"x" * (MAX_UPLOAD_BYTES + 1)),
        filename="resume.pdf",
        content_type="application/pdf",
    )

    with app.app_context(), pytest.raises(ValueError, match="smaller than 4 MB"):
        store_resume(upload, user_id=42)
