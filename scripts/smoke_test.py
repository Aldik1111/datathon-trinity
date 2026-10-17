"""Actual HTTP + spawned worker smoke check, using generated geometry only.

Run: python scripts/smoke_test.py. No uploads or frames are retained.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ui.client import APISettings, get_job, submit_video


def synthetic_mov(path: Path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (96, 64))
    if not writer.isOpened():
        raise RuntimeError('MOV fixture codec is unavailable')
    try:
        for index in range(10):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            cv2.rectangle(frame, (index * 5, 10), (index * 5 + 20, 30), (50, 200, 100), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path.read_bytes()


@contextmanager
def running_server(folder: Path, mode='demo', analyzer='src.vision.analyzer:analyze_video'):
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    folder.mkdir(parents=True, exist_ok=True)
    settings = APISettings(f'http://127.0.0.1:{port}', secrets.token_urlsafe(32))
    env = dict(os.environ, API_KEY=settings.api_key, UPLOAD_DIR=str(folder/'uploads'),
               API_DB_PATH=str(folder/'jobs.sqlite3'), ANALYZER_MODE=mode,
               ANALYZER_CALLABLE=analyzer, ANALYZER_TIMEOUT_SEC='20', MAX_UPLOAD_MB='2',
               MAX_PENDING_JOBS='2', JOB_TTL_SEC='60', RULES_PATH=str(ROOT/'configs/rules.json'),
               PYTHONPATH=str(ROOT), STREAMLIT_BROWSER_GATHER_USAGE_STATS='false')
    process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'src.api.app:create_app', '--factory', '--host',
         '127.0.0.1', '--port', str(port), '--workers', '1', '--no-access-log'],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
    )
    try:
        deadline = time.monotonic()+25
        while True:
            if process.poll() is not None:
                raise RuntimeError('Smoke API exited before becoming healthy')
            try:
                if requests.get(settings.base_url+'/healthz', timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Smoke API startup timed out')
            time.sleep(.1)
        yield settings
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def check(folder: Path, mode: str) -> dict:
    with running_server(folder, mode=mode) as settings:
        video = synthetic_mov(folder/'generated.mov')
        assert requests.get(settings.base_url+'/api/v1/jobs/unknown', timeout=3).status_code == 401
        assert requests.post(settings.base_url+'/api/v1/analyze', files={'file': ('generated.mov', video)}, timeout=3).status_code == 401
        response = submit_video(settings, 'generated.mov', video)
        assert response['status'] == 'processing'
        deadline = time.monotonic()+25
        while True:
            result = get_job(settings, response['job_id'])
            if result['status'] != 'processing':
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Smoke analysis timed out')
            time.sleep(.1)
        if mode == 'demo':
            assert result['status'] == 'done' and result['demo'] is True
            assert result['verdict'] == 'violation'
            for event in result['violations']:
                assert event['rule_status'] == 'unverified' and event['rule'] is None
                assert 1 <= len(event['evidence_frames']) <= 3
                assert 0 <= event['timecode_sec'] <= 1
                for frame in event['evidence_frames']:
                    image = base64.b64decode(frame['image_base64'], validate=True)
                    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
                    assert decoded is not None and 0 <= frame['timestamp_sec'] <= 1
        else:
            assert result['status'] == 'failed' and result['demo'] is False
            assert 'verdict' not in result and 'Local model is not installed' in result['error']
        # Status commit can precede unlink by one scheduler turn.
        for _ in range(30):
            if not list((folder/'uploads').glob('*.mov')):
                break
            time.sleep(.1)
        assert not list((folder/'uploads').glob('*.mov'))
        return {'mode': mode, 'status': result['status'], 'uploads_cleaned': True,
                'synthetic_only': True, 'recognition_quality_checked': False}


def main():
    with tempfile.TemporaryDirectory(prefix='trinity-synthetic-') as directory:
        folder = Path(directory)
        print(json.dumps([check(folder/'demo', 'demo'), check(folder/'real', 'real')], indent=2))


if __name__ == '__main__':
    main()
