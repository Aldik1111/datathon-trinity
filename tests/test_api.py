import importlib
import json
import sqlite3
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from src.api.settings import Settings

api_module = importlib.import_module("src.api.app")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(
            api_key="test-secret",
            upload_dir=Path(self.temp.name) / "uploads",
            db_path=Path(self.temp.name) / "jobs.sqlite3",
            max_upload_mb=1,
            analyzer_mode="demo",
            analyzer_timeout_sec=3,
        )
        self.client = TestClient(api_module.create_app(self.settings))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_api_key_required_for_both_routes(self):
        upload = self.client.post(
            "/api/v1/analyze",
            files={"file": ("clip.mov", b"x" * (2 * 1024 * 1024), "video/quicktime")},
        )
        self.assertEqual(upload.status_code, 401)
        status = self.client.get("/api/v1/jobs/unknown")
        self.assertEqual(status.status_code, 401)
        readiness = self.client.get("/api/v1/readiness")
        self.assertEqual(readiness.status_code, 401)

    def test_readiness_reports_default_routing_and_jury_compatibility(self):
        response = self.client.get("/api/v1/readiness", headers={"X-API-Key": "test-secret"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["default_analysis_type"], "jury")
        self.assertEqual(data["omitted_analysis_type_routes_to"], "jury")
        self.assertFalse(data["jury_contract_compatible"])
        self.assertEqual(data["maximum_analysis_deadline_sec"], 300)

    def test_jury_default_rejects_known_experimental_analyzer(self):
        settings = Settings(
            api_key="test-secret",
            upload_dir=Path(self.temp.name) / "experimental-uploads",
            db_path=Path(self.temp.name) / "experimental.sqlite3",
            max_upload_mb=1,
            analyzer_mode="real",
            analyzer_callable="src.vision.scenario_classifier:analyze_video",
            default_analysis_type="jury",
        )
        with TestClient(api_module.create_app(settings)) as client:
            response = client.post(
                "/api/v1/analyze",
                headers={"X-API-Key": "test-secret"},
                files={"file": ("clip.mov", b"synthetic", "video/quicktime")},
            )
            self.assertEqual(response.status_code, 409)
            self.assertIn("experimental", response.json()["detail"])

    def test_demo_ignores_real_analyzer_choice_and_remains_synthetic(self):
        for index, default in enumerate((None, "jury")):
            with self.subTest(default=default):
                settings = Settings(api_key="test-secret", max_upload_mb=1,
                    upload_dir=Path(self.temp.name) / f"demo-uploads-{index}",
                    analyzer_mode="demo", default_analysis_type=default,
                    analyzer_callable="src.vision.safe_classifier:analyze_video")
                with patch.object(api_module, "_run_child", return_value={
                    "verdict": "no_violation", "violations": [], "demo": True}), \
                        TestClient(api_module.create_app(settings)) as client:
                    headers = {"X-API-Key": "test-secret"}
                    readiness = client.get("/api/v1/readiness", headers=headers).json()
                    self.assertEqual(readiness["omitted_analysis_type_routes_to"], "jury")
                    self.assertFalse(readiness["jury_contract_compatible"])
                    response = client.post("/api/v1/analyze", headers=headers,
                                           files={"file": ("synthetic.mov", b"synthetic")})
                    self.assertEqual(response.status_code, 202)
                    self.assertTrue(response.json()["demo"])

    def test_missing_legacy_analyzer_is_reported_as_unverified_not_ready(self):
        settings = Settings(
            api_key="test-secret",
            upload_dir=Path(self.temp.name) / "missing-uploads",
            db_path=Path(self.temp.name) / "missing.sqlite3",
            max_upload_mb=1,
            analyzer_mode="real",
            analyzer_callable="src.vision.analyzer:analyze_video",
        )
        with TestClient(api_module.create_app(settings)) as client:
            readiness = client.get("/api/v1/readiness", headers={"X-API-Key": "test-secret"}).json()
        self.assertTrue(readiness["configured_for_jury_contract"])
        self.assertIsNone(readiness["jury_contract_compatible"])
        self.assertFalse(readiness["model_load_verified"])

    def test_explicit_experimental_default_is_not_jury_compatible_for_custom_analyzer(self):
        settings = Settings(
            api_key="test-secret",
            upload_dir=Path(self.temp.name) / "explicit-scenario-uploads",
            db_path=Path(self.temp.name) / "explicit-scenario.sqlite3",
            max_upload_mb=1,
            analyzer_mode="real",
            analyzer_callable="custom.analyzer:analyze_video",
            default_analysis_type="bank_scenarios",
        )
        with TestClient(api_module.create_app(settings)) as client:
            readiness = client.get("/api/v1/readiness", headers={"X-API-Key": "test-secret"}).json()
        self.assertEqual(readiness["default_analysis_type"], "bank_scenarios")
        self.assertFalse(readiness["jury_contract_compatible"])

    def test_omitted_scenario_mode_preserves_configured_analyzer_and_model_path(self):
        settings = Settings(
            api_key="test-secret",
            upload_dir=Path(self.temp.name) / "scenario-uploads",
            db_path=Path(self.temp.name) / "scenario.sqlite3",
            max_upload_mb=1,
            analyzer_mode="real",
            analyzer_callable="src.vision.scenario_classifier:analyze_video",
            model_path="legacy-scenario-model",
        )
        observed = []

        def fake_analysis(path, configured, timeout_sec, stop_event=None):
            selection_path = Path(str(path) + ".analysis.json")
            mode_path = Path(str(path) + ".api-mode.json")
            observed.append((selection_path.exists(), mode_path.read_text(encoding="utf-8")))
            return {"verdict": "no_candidate", "violations": [], "analysis_scope": "bank_scenarios"}

        original = api_module._run_child
        api_module._run_child = fake_analysis
        try:
            with TestClient(api_module.create_app(settings)) as client:
                accepted = client.post(
                    "/api/v1/analyze",
                    headers={"X-API-Key": "test-secret"},
                    files={"file": ("clip.mov", b"synthetic", "video/quicktime")},
                )
                self.assertEqual(accepted.status_code, 202)
                job_id = accepted.json()["job_id"]
                result = None
                deadline = time.time() + 2
                while time.time() < deadline:
                    result = client.get(
                        f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "test-secret"}
                    ).json()
                    if result["status"] in {"done", "failed"}:
                        break
                    time.sleep(0.02)
                self.assertEqual(result["status"], "done")
                self.assertEqual(result["verdict"], "no_candidate")
        finally:
            api_module._run_child = original
        self.assertEqual(observed[0][0], False)
        self.assertEqual(json.loads(observed[0][1]), {"analysis_type": "bank_scenarios"})

    def test_rejects_non_mov_and_oversize_uploads(self):
        headers = {"X-API-Key": "test-secret"}
        wrong_type = self.client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("clip.mp4", b"data", "video/mp4")},
        )
        self.assertEqual(wrong_type.status_code, 415)
        too_large = self.client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("clip.mov", b"x" * (1024 * 1024 + 1), "video/quicktime")},
        )
        self.assertEqual(too_large.status_code, 413)

    def test_job_persists_and_returns_exact_done_contract(self):
        def fake_analysis(path, settings, timeout_sec, stop_event=None):
            self.assertTrue(Path(path).name.endswith(".mov"))
            return {
                "verdict": "violation",
                "violations": [
                    {
                        "type": "unsafe_action",
                        "timecode_sec": 2.5,
                        "description": "Example violation",
                        "rule": {"rule_id": "R-1", "title": "Safe rule"},
                        "rule_status": "verified",
                        "evidence_frames": [
                            {"timestamp_sec": 2.5, "image_base64": "ZmFrZQ=="}
                        ],
                    }
                ],
                "demo": True,
            }

        original = api_module._run_child
        api_module._run_child = fake_analysis
        self.addCleanup(setattr, api_module, "_run_child", original)
        headers = {"X-API-Key": "test-secret"}
        accepted = self.client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("scene.MOV", b"video-bytes", "video/quicktime")},
        )
        self.assertEqual(accepted.status_code, 202)
        job_id = accepted.json()["job_id"]
        self.assertEqual(accepted.json()["status"], "processing")

        result = None
        deadline = time.time() + 2
        while time.time() < deadline:
            response = self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            if result["status"] in {"done", "failed"}:
                break
            time.sleep(0.02)

        self.assertEqual(
            result,
            {
                "job_id": job_id,
                "status": "done",
                "verdict": "violation",
                "violations": [
                    {
                        "type": "unsafe_action",
                        "timecode_sec": 2.5,
                        "description": "Example violation",
                        "rule": {"rule_id": "R-1", "title": "Safe rule"},
                        "rule_status": "verified",
                        "evidence_frames": [
                            {"timestamp_sec": 2.5, "image_base64": "ZmFrZQ=="}
                        ],
                    }
                ],
                "demo": True,
            },
        )

    def test_concurrent_uploads_reserve_capacity_without_orphaning_a_job(self):
        settings = Settings(
            api_key="test-secret",
            upload_dir=Path(self.temp.name) / "limited-uploads",
            db_path=Path(self.temp.name) / "limited.sqlite3",
            max_upload_mb=1,
            analyzer_mode="demo",
            max_pending_jobs=1,
        )
        app = api_module.create_app(settings)
        original = api_module._run_child

        def slow_analysis(path, settings, timeout_sec, stop_event=None):
            time.sleep(0.75)
            return {"verdict": "no_violation", "violations": [], "demo": True}

        api_module._run_child = slow_analysis
        try:
            with TestClient(app) as client:
                def post_one(_):
                    return client.post(
                        "/api/v1/analyze",
                        headers={"X-API-Key": "test-secret"},
                        files={"file": ("clip.mov", b"x" * 128_000, "video/quicktime")},
                    ).status_code

                with ThreadPoolExecutor(max_workers=2) as pool:
                    statuses = list(pool.map(post_one, range(2)))
                self.assertCountEqual(statuses, [202, 503])
                db = sqlite3.connect(settings.database_path)
                try:
                    jobs = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                finally:
                    db.close()
                self.assertEqual(jobs, 1)
        finally:
            api_module._run_child = original

    def test_expiry_keeps_active_jobs_and_restart_fails_them_and_removes_upload(self):
        interrupted_path = self.settings.upload_dir / "interrupted.mov"
        active_path = self.settings.upload_dir / "active.mov"
        interrupted_path.write_bytes(b"old upload")
        active_path.write_bytes(b"active upload")
        old = time.time() - 10_000
        db = sqlite3.connect(self.settings.database_path)
        try:
            db.execute(
                "INSERT INTO jobs(job_id,status,video_path,created_at,updated_at) VALUES(?,?,?,?,?)",
                ("interrupted", "processing", str(interrupted_path), old, old),
            )
            db.execute(
                "INSERT INTO jobs(job_id,status,video_path,created_at,updated_at) VALUES(?,?,?,?,?)",
                ("active", "processing", str(active_path), old, old),
            )
            db.commit()
        finally:
            db.close()
        api_module._cleanup_expired(self.settings)
        db = sqlite3.connect(self.settings.database_path)
        try:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM jobs WHERE status='processing'").fetchone()[0],
                2,
            )
        finally:
            db.close()
        # Startup treats persisted processing rows as interrupted and removes their uploads.
        api_module._initialize(self.settings)
        db = sqlite3.connect(self.settings.database_path)
        try:
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM jobs WHERE status='failed'").fetchone()[0],
                2,
            )
        finally:
            db.close()
        self.assertFalse(interrupted_path.exists())
        self.assertFalse(active_path.exists())


if __name__ == "__main__":
    unittest.main()
