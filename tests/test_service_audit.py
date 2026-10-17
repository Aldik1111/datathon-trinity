"""Regression checks using only synthetic bytes and temporary state."""
import asyncio
import json
import sqlite3
import time
import uuid
from dataclasses import replace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.api.app import ApiRequestGuard, _cleanup_expired, _connect, _initialize, create_app
from src.api.settings import Settings
from src.runtime_config import read_api_key


@pytest.fixture
def settings(tmp_path):
    return Settings(api_key="synthetic-secret", upload_dir=tmp_path / "uploads", max_upload_mb=1, analyzer_mode="demo", job_ttl_sec=1)


def test_auth_rejects_before_reading_body():
    async def scenario():
        async def forbidden(*args):
            pytest.fail("Unauthenticated request reached body or application")
        messages = []
        async def send(message):
            messages.append(message)
        guard = ApiRequestGuard(forbidden, "key", 100)
        await guard({"type": "http", "path": "/api/v1/analyze", "method": "POST", "headers": []}, forbidden, send)
        assert messages[0]["status"] == 401
    asyncio.run(scenario())


def test_chunked_body_is_limited_without_content_length():
    async def scenario():
        async def app(*args):
            pytest.fail("Oversized body reached multipart parser")
        messages = []
        async def receive():
            return {"type": "http.request", "body": b"x" * 70000, "more_body": True}
        async def send(message):
            messages.append(message)
        guard = ApiRequestGuard(app, "key", 100)
        await guard({"type": "http", "path": "/api/v1/analyze", "method": "POST", "headers": [(b"x-api-key", b"key")]}, receive, send)
        assert messages[0]["status"] == 413
    asyncio.run(scenario())


def test_expiry_removes_result_and_file_but_retains_active(settings):
    _initialize(settings)
    expired = settings.upload_dir / "expired.mov"
    expired.write_bytes(b"synthetic")
    with _connect(settings) as db:
        assert db.execute("PRAGMA secure_delete").fetchone()[0] == 1
        for job_id, status in [("expired", "done"), ("active", "processing")]:
            db.execute("INSERT INTO jobs(job_id,status,video_path,created_at,updated_at,result_json) VALUES(?,?,?,?,?,?)", (job_id, status, str(expired), 0, 0, '{"synthetic":"evidence"}'))
    _cleanup_expired(settings)
    with _connect(settings) as db:
        assert [r[0] for r in db.execute("SELECT job_id FROM jobs")] == ["active"]
    assert not expired.exists()


def test_startup_removes_generated_orphans_only(settings):
    settings.upload_dir.mkdir()
    generated = [settings.upload_dir / f"{uuid.uuid4()}{suffix}" for suffix in (".mov", ".worker.json")]
    for path in generated:
        path.write_bytes(b"synthetic orphan")
    other = settings.upload_dir / "user-notes.txt"
    other.write_text("keep")
    _initialize(settings)
    assert not any(path.exists() for path in generated)
    assert other.read_text() == "keep"


def test_unknown_job_and_wrong_key(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/api/v1/jobs/missing", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/v1/jobs/missing", headers={"X-API-Key": settings.api_key}).status_code == 404
        assert client.post("/api/v1/analyze", headers={"X-API-Key": settings.api_key}, files={"file": ("empty.mov", b"")}).status_code == 400
        assert list(settings.upload_dir.glob("*.mov")) == []


def test_deadline_cannot_exceed_jury_limit(settings):
    with pytest.raises(ValueError):
        replace(settings, analyzer_timeout_sec=301)


def test_file_secret_and_invalid_config_are_not_echoed(tmp_path, monkeypatch):
    secret = tmp_path / "api.key"
    secret.write_text("synthetic-key\n")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_KEY_FILE", str(secret))
    assert read_api_key() == "synthetic-key"
    monkeypatch.setenv("API_KEY", "another-secret")
    with pytest.raises(ValueError, match="either") as error:
        read_api_key()
    assert "another-secret" not in str(error.value)


def test_demo_is_not_presented_as_detection():
    at = AppTest.from_string('''
from src.ui.app import _render_result
_render_result({"demo": True, "verdict": "violation", "violations": []})
''').run()
    assert not at.exception
    assert not at.error
    assert any("распознавание нарушений не выполнялось" in item.value for item in at.info)


def test_unverified_rule_is_not_rendered_as_source():
    at = AppTest.from_string('''
from src.ui.app import _render_violation
_render_violation({"type":"synthetic", "rule_status":"unverified", "rule":{"title":"UNVERIFIED_SOURCE","url":"https://example.test"}, "evidence_frames":[]}, 1)
''').run()
    assert not at.exception
    assert any("не подтверждено" in item.value for item in at.warning)
    assert not any("UNVERIFIED_SOURCE" in item.value for item in at.markdown)


def test_missing_job_stops_ui_polling():
    from src.ui.client import APIClientError
    at = AppTest.from_string('''
import streamlit as st
from src.ui.app import _job_fragment
st.session_state.job_id = "expired"
_job_fragment()
''')
    with patch("src.ui.app.get_job", side_effect=APIClientError("not found", 404)):
        at.run()
    assert not at.exception
    assert at.session_state.job_status == "failed"


def test_child_does_not_echo_model_logs_or_exceptions(settings, capfd, tmp_path):
    import cv2
    import numpy as np
    from src.api.app import _run_child, SafeAnalysisFailure
    video = tmp_path / "synthetic.mov"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    assert writer.isOpened()
    writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    configured = replace(settings, analyzer_mode="real", analyzer_callable="tests.fake_analyzers:noisy_failure", rules_path=None)
    with pytest.raises(SafeAnalysisFailure) as error:
        _run_child(str(video), configured, 10)
    captured = capfd.readouterr()
    assert "SYNTHETIC_PRIVATE_MARKER" not in captured.out + captured.err + str(error.value)
    assert not video.with_suffix(".worker.json").exists()


def test_deployment_defaults_are_local_and_secret_files_are_not_image_inputs():
    from pathlib import Path
    config = json.loads(Path("deploy/compose.json").read_text())
    assert config["networks"]["backend"]["internal"] is True
    for service in config["services"].values():
        assert all(port.startswith("127.0.0.1:") for port in service["ports"])
        assert service["read_only"] is True
        assert "API_KEY" not in service["environment"]
        assert service["secrets"] == ["api_key"]
    public = json.loads(Path("deploy/compose.public.json").read_text())
    assert public["services"]["gateway"]["secrets"] == ["ui_users"]
    dockerfile = Path("Dockerfile").read_text()
    assert "COPY . " not in dockerfile
    assert "USER service" in dockerfile
