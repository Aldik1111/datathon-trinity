"""End-to-end checks for the local spawned analysis worker and SQLite state."""

from __future__ import annotations

import base64
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import Settings


class ServiceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.upload_dir = self.root / "uploads"
        self.db_path = self.root / "jobs.sqlite3"
        self.video_bytes = self._make_video()
        self.headers = {"X-API-Key": "integration-secret"}

    def _make_video(self) -> bytes:
        video_path = self.root / "fixture.mov"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 48)
        )
        if not writer.isOpened():
            self.skipTest("OpenCV cannot create the MOV fixture in this environment")
        for index in range(5):
            frame = np.full((48, 64, 3), 20 + index * 30, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return video_path.read_bytes()

    def _settings(
        self,
        *,
        mode: str = "demo",
        analyzer: str = "tests.fake_analyzers:no_violation",
        timeout: int = 10,
    ) -> Settings:
        return Settings(
            api_key="integration-secret",
            upload_dir=self.upload_dir,
            db_path=self.db_path,
            max_upload_mb=2,
            analyzer_mode=mode,
            analyzer_callable=analyzer,
            rules_path=None,
            analyzer_timeout_sec=timeout,
            max_pending_jobs=2,
        )

    def _submit(self, client: TestClient, payload: bytes | None = None) -> str:
        response = client.post(
            "/api/v1/analyze",
            headers=self.headers,
            files={
                "file": (
                    "fixture.mov",
                    self.video_bytes if payload is None else payload,
                    "video/quicktime",
                )
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()["job_id"]

    def _wait_for_job(self, client: TestClient, job_id: str, timeout: float = 12) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = client.get(f"/api/v1/jobs/{job_id}", headers=self.headers)
            self.assertEqual(response.status_code, 200, response.text)
            job = response.json()
            if job["status"] in {"done", "failed"}:
                return job
            time.sleep(0.05)
        self.fail(f"Job {job_id} did not finish within {timeout} seconds")

    def test_spawned_demo_worker_returns_marked_valid_jpeg_and_survives_restart(self) -> None:
        settings = self._settings(mode="demo")
        with TestClient(create_app(settings)) as client:
            job_id = self._submit(client)
            result = self._wait_for_job(client, job_id)
            self.assertEqual(result["status"], "done")
            self.assertTrue(result["demo"])
            self.assertEqual(result["verdict"], "violation")
            self.assertEqual(len(result["violations"]), 1)
            frames = result["violations"][0]["evidence_frames"]
            self.assertEqual(len(frames), 1)
            jpeg = base64.b64decode(frames[0]["image_base64"], validate=True)
            self.assertTrue(jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9"))

        # A new app instance over the same SQLite file retains completed results.
        with TestClient(create_app(settings)) as restarted:
            recovered = restarted.get(
                f"/api/v1/jobs/{job_id}", headers=self.headers
            )
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual(recovered.json(), result)

    def test_real_mode_missing_analyzer_and_corrupt_video_fail(self) -> None:
        missing = self._settings(
            mode="real", analyzer="tests.no_such_model:analyze_video"
        )
        with TestClient(create_app(missing)) as client:
            job_id = self._submit(client)
            result = self._wait_for_job(client, job_id)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["demo"])
            self.assertNotIn("verdict", result)

        corrupt = self._settings(mode="demo")
        with TestClient(create_app(corrupt)) as client:
            job_id = self._submit(client, b"this is not a video")
            result = self._wait_for_job(client, job_id)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("verdict", result)

    def test_real_no_violation_comes_from_analyzer_and_api_stays_responsive(self) -> None:
        settings = self._settings(mode="real")
        with TestClient(create_app(settings)) as client:
            job_id = self._submit(client)
            started = time.monotonic()
            response = client.get(f"/api/v1/jobs/{job_id}", headers=self.headers)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertEqual(response.status_code, 200)

            result = self._wait_for_job(client, job_id)
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["verdict"], "no_violation")
            self.assertEqual(result["violations"], [])
            self.assertFalse(result["demo"])

    def test_spawned_analyzer_is_terminated_at_job_deadline(self) -> None:
        settings = self._settings(
            mode="real", analyzer="tests.fake_analyzers:wait_then_no_violation", timeout=1
        )
        with TestClient(create_app(settings)) as client:
            job_id = self._submit(client)
            result = self._wait_for_job(client, job_id, timeout=8)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"], "Analysis timed out")


if __name__ == "__main__":
    unittest.main()
