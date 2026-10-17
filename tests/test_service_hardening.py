"""Lifecycle and transport regressions; fixtures contain no external data."""
import asyncio
import importlib
from pathlib import Path
import sqlite3
import time
import uuid

from fastapi.testclient import TestClient
import pytest

from src.api.settings import Settings
from src.api.runtime import instance_lock
from scripts.smoke_test import check

api = importlib.import_module('src.api.app')


def settings(tmp_path, **changes):
    return Settings(api_key='synthetic-test-key', upload_dir=tmp_path/'uploads',
                    db_path=tmp_path/'jobs.sqlite3', max_upload_mb=1, analyzer_mode='demo', **changes)


def test_instance_lock_prevents_second_api_and_releases(tmp_path):
    configured = settings(tmp_path)
    with TestClient(api.create_app(configured)) as client:
        assert client.get('/healthz').json() == {'status': 'ok'}
        with pytest.raises(RuntimeError, match='already has an API instance'):
            with TestClient(api.create_app(configured)):
                pass
    with instance_lock(configured.database_path):
        pass


def test_restart_removes_service_orphans_only(tmp_path):
    configured = settings(tmp_path)
    configured.upload_dir.mkdir()
    owned = configured.upload_dir/(str(uuid.uuid4())+'.mov')
    other = configured.upload_dir/'manually-created.mov'
    owned.write_bytes(b'synthetic abandoned upload')
    other.write_bytes(b'synthetic unrelated file')
    api._initialize(configured)
    assert not owned.exists() and other.exists()


def test_waiting_job_expires_without_deleting_active_or_external_files(tmp_path):
    configured = settings(tmp_path, analyzer_timeout_sec=1)
    api._initialize(configured)
    paths = [configured.upload_dir/'waiting.mov', configured.upload_dir/'active.mov', tmp_path/'external.mov']
    for path in paths:
        path.write_bytes(b'synthetic')
    with sqlite3.connect(configured.database_path) as db:
        for key, path in zip(['waiting', 'active', 'external'], paths):
            db.execute('INSERT INTO jobs(job_id,status,video_path,created_at,updated_at) VALUES(?,?,?,?,?)',
                       (key, 'processing', str(path), time.time()-2, time.time()-2))
    api._expire_waiting(configured, 'active')
    assert not paths[0].exists() and paths[1].exists() and paths[2].exists()
    with sqlite3.connect(configured.database_path) as db:
        rows = dict(db.execute('SELECT job_id,status FROM jobs'))
    assert rows == {'waiting': 'failed', 'active': 'processing', 'external': 'failed'}


def test_chunked_upload_guard_rejects_without_parsing(tmp_path):
    called = False
    async def target(scope, receive, send):
        nonlocal called
        called = True
    messages = iter([{'type': 'http.request', 'body': b'x' * 70000, 'more_body': True},
                     {'type': 'http.request', 'body': b'x' * 70000, 'more_body': False}])
    sent = []
    async def receive():
        return next(messages)
    async def send(message):
        sent.append(message)
    app = api.ApiRequestGuard(target, 'key', 100)
    asyncio.run(app({'type': 'http', 'method': 'POST', 'path': '/api/v1/analyze',
                     'headers': [(b'x-api-key', b'key')]}, receive, send))
    assert sent[0]['status'] == 413 and not called


def test_actual_http_and_spawned_worker_synthetic_only(tmp_path):
    assert check(tmp_path/'demo', 'demo')['status'] == 'done'
    assert check(tmp_path/'real', 'real')['status'] == 'failed'


@pytest.mark.parametrize('timeout', [0, 301])
def test_timeout_configuration_is_capped(tmp_path, timeout):
    with pytest.raises(ValueError, match='between 1 and 300'):
        settings(tmp_path, analyzer_timeout_sec=timeout)
