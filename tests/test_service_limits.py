import asyncio
from dataclasses import replace
from pathlib import Path
import sqlite3
import time

import pytest

from src.api.app import ApiRequestGuard, _cleanup_expired, _initialize
from src.api.settings import Settings


def settings(tmp_path):
    return Settings(api_key="synthetic-key", upload_dir=tmp_path / "uploads",
                    db_path=tmp_path / "jobs.sqlite3", max_upload_mb=1, job_ttl_sec=1)


@pytest.mark.parametrize("timeout", [0, 301])
def test_timeout_cannot_exceed_contract(tmp_path, timeout):
    with pytest.raises(ValueError, match="between 1 and 300"):
        replace(settings(tmp_path), analyzer_timeout_sec=timeout)


def test_ttl_removes_completed_result_and_upload(tmp_path):
    config = settings(tmp_path)
    _initialize(config)
    path = config.upload_dir / "expired.mov"
    path.write_bytes(b"synthetic")
    old = time.time() - 5
    with sqlite3.connect(config.database_path) as db:
        db.execute("INSERT INTO jobs(job_id,status,video_path,created_at,updated_at,result_json) VALUES(?,?,?,?,?,?)",
                   ("expired", "done", str(path), old, old, '{"demo": true}'))
    _cleanup_expired(config)
    with sqlite3.connect(config.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert not path.exists()


@pytest.mark.parametrize("authorized,expected", [(False, 401), (True, 413)])
def test_guard_rejects_before_parser_and_caps_chunked_body(authorized, expected):
    called, messages, received = [], [], []

    async def downstream(scope, receive, send):
        called.append(True)

    async def receive():
        received.append(True)
        return {"type": "http.request", "body": b"x" * 65536, "more_body": True}

    async def send(message):
        messages.append(message)

    scope = {"type": "http", "path": "/api/v1/analyze", "method": "POST",
             "headers": [(b"x-api-key", b"synthetic-key")] if authorized else []}
    asyncio.run(ApiRequestGuard(downstream, "synthetic-key", 1)(scope, receive, send))
    assert not called
    assert messages[0]["status"] == expected
    assert len(received) == (2 if authorized else 0)
