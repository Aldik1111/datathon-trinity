"""Real loopback HTTP -> SQLite queue -> spawned analyzer; synthetic MOV only."""
import base64
from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / "synthetic.mov"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    assert writer.isOpened(), "Synthetic MOV encoder is required for this check"
    try:
        for index in range(5):
            writer.write(np.full((48, 64, 3), index * 40, dtype=np.uint8))
    finally:
        writer.release()
    return path.read_bytes()


@contextmanager
def server(tmp_path, mode="demo", analyzer="tests.missing_analyzer:analyze_video"):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_urlsafe(32)
    env = dict(os.environ, API_KEY=key, ANALYZER_MODE=mode,
               ANALYZER_CALLABLE=analyzer,
               UPLOAD_DIR=str(tmp_path / "uploads"), API_DB_PATH=str(tmp_path / "jobs.sqlite3"),
               MAX_UPLOAD_MB="1", MAX_PENDING_JOBS="2", ANALYZER_TIMEOUT_SEC="10",
               JOB_TTL_SEC="86400", RULES_PATH=str(ROOT / "configs/rules.json"),
               PYTHON_DOTENV_DISABLED="1")
    creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (tmp_path / "server.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "src.api.app:create_app", "--factory",
             "--host", "127.0.0.1", "--port", str(port), "--workers", "1", "--no-access-log"],
            cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=creation,
        )
        session = requests.Session()
        session.trust_env = False
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                assert process.poll() is None, "API exited during startup"
                try:
                    if session.get(url + "/healthz", timeout=1).status_code == 200:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(.1)
            else:
                pytest.fail("API did not start")
            yield session, url, {"X-API-Key": key}
        finally:
            session.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    assert key not in (tmp_path / "server.log").read_text(errors="replace")


def wait_job(session, url, headers, job_id):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = session.get(url + f"/api/v1/jobs/{job_id}", headers=headers, timeout=2)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in {"processing", "done", "failed"}
        if result["status"] != "processing":
            return result
        time.sleep(.05)
    pytest.fail("Job exceeded test deadline")


def test_http_demo_contract_restart_and_orphan_cleanup(tmp_path, movie):
    with server(tmp_path) as (session, url, headers):
        for route in ("/api/v1/analyze", "/api/v1/jobs/unknown"):
            response = (session.post(url + route, data=b"invalid", timeout=2) if route.endswith("analyze")
                        else session.get(url + route, timeout=2))
            assert response.status_code == 401
        accepted = session.post(url + "/api/v1/analyze", headers=headers,
                                files={"file": ("synthetic.mov", movie, "video/quicktime")}, timeout=5)
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]
        result = wait_job(session, url, headers, job_id)
        assert result["status"] == "done" and result["demo"] is True
        assert result["verdict"] == "violation"
        event = result["violations"][0]
        assert {"type", "timecode_sec", "description", "evidence_frames"} <= event.keys()
        assert event["rule_status"] == "unverified" and event["rule"] is None
        assert 1 <= len(event["evidence_frames"]) <= 3
        for frame in event["evidence_frames"]:
            assert 0 <= frame["timestamp_sec"] <= 1
            jpeg = base64.b64decode(frame["image_base64"], validate=True)
            assert cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) is not None
        # Cleanup may follow the DB commit by a few milliseconds.
        deadline = time.monotonic() + 2
        while list((tmp_path / "uploads").glob("*.mov")) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not list((tmp_path / "uploads").glob("*.mov"))
        oversized = session.post(url + "/api/v1/analyze", headers=headers,
                                 files={"file": ("synthetic.mov", b"x" * (1024**2 + 1))}, timeout=5)
        assert oversized.status_code == 413
    orphan = tmp_path / "uploads" / f"{uuid.uuid4()}.mov"
    orphan.write_bytes(movie)
    interrupted = tmp_path / "uploads" / f"{uuid.uuid4()}.mov"
    interrupted.write_bytes(movie)
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        db.execute("INSERT INTO jobs(job_id,status,video_path,created_at,updated_at) VALUES(?,?,?,?,?)",
                   ("interrupted", "processing", str(interrupted), time.time(), time.time()))
    with server(tmp_path) as (session, url, headers):
        assert wait_job(session, url, headers, job_id) == result
        failed = wait_job(session, url, headers, "interrupted")
        assert failed["status"] == "failed" and "restart" in failed["error"]
        assert not orphan.exists() and not interrupted.exists()


def test_http_real_without_model_fails_clearly(tmp_path, movie):
    with server(tmp_path, "real") as (session, url, headers):
        accepted = session.post(url + "/api/v1/analyze", headers=headers,
                                files={"file": ("synthetic.mov", movie)}, timeout=5)
        assert accepted.status_code == 202
        result = wait_job(session, url, headers, accepted.json()["job_id"])
        assert result["status"] == "failed" and result["demo"] is False
        assert "Local model is not installed" in result["error"]
        assert "verdict" not in result


def test_model_output_and_exception_never_reach_logs(tmp_path, movie):
    with server(tmp_path, "real", "tests.fake_analyzers:noisy_failure") as (session, url, headers):
        accepted = session.post(url + "/api/v1/analyze", headers=headers,
                                files={"file": ("synthetic.mov", movie)}, timeout=5)
        assert accepted.status_code == 202
        result = wait_job(session, url, headers, accepted.json()["job_id"])
        assert result["status"] == "failed"
        assert result["error"] == "Local model failed to analyze the video"
    assert "SYNTHETIC_PRIVATE_CANARY" not in (tmp_path / "server.log").read_text(errors="replace")
