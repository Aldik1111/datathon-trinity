"""Routing/lifecycle tests; fake workers here are not real-model validation."""
import asyncio
import base64
import importlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.settings import Settings, get_settings

api = importlib.import_module("src.api.app")
HEADERS = {"X-API-Key": "jury-test-key"}


def configuration(tmp_path, **overrides):
    model = tmp_path / "scenario-model"
    model.mkdir(exist_ok=True)
    for name in ("metadata.json", "model.npz", "backbone.onnx"):
        (model / name).write_bytes(b"test fixture, not real weights")
    values = dict(api_key=HEADERS["X-API-Key"], upload_dir=tmp_path / "uploads",
                  db_path=tmp_path / "jobs.sqlite3", max_upload_mb=1,
                  analyzer_mode="real", default_analysis_type="jury", jury_analyzer="bank_scenarios",
                  analyzer_callable="src.vision.safe_classifier:analyze_video",
                  model_path="separate-safe-model", scenario_model_path=str(model),
                  rules_path=None, max_pending_jobs=4)
    values.update(overrides)
    return Settings(**values)


def projected_result(positive=False):
    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return {
        "verdict": "violation" if positive else "no_violation", "demo": False,
        "analysis_scope": "bank_scenarios", "output_contract": "jury-v1",
        "jury_policy": "scenario-threshold-v1",
        "internal_verdict": "review_required" if positive else "no_candidate",
        "provenance": {"actual_analyzer": "src.vision.scenario_classifier:analyze_video"},
        "coverage": {"all_classes_supported": True},
        "video_duration_sec": 1.0, "rules_status": "unverified",
        "violations": [{"type": "money_returned", "timecode_sec": 0.5,
                        "description": "Model candidate requiring review.",
                        "evidence_frames": [{"timestamp_sec": 0.5,
                            "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")}]}] if positive else [],
    }


def submit(client, **fields):
    return client.post("/api/v1/analyze", headers=HEADERS,
                       files={"file": ("fixture.mov", b"synthetic routing fixture", "video/quicktime")},
                       data=fields)


def terminal(client, job_id):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS)
        assert response.status_code == 200
        result = response.json()
        if result["status"] != "processing":
            return result
        time.sleep(.01)
    pytest.fail("Job did not reach a terminal state")


@pytest.mark.parametrize("positive", [False, True])
def test_omitted_mode_uses_scenarios_and_preserves_jury_metadata(tmp_path, monkeypatch, positive):
    configured = configuration(tmp_path)
    seen = []

    def worker(path, settings, timeout_sec, stop_event=None):
        seen.append(path)
        assert json.loads(Path(path + ".analysis.json").read_text()) == {"analysis_type": "bank_scenarios"}
        assert json.loads(Path(path + ".api-mode.json").read_text()) == {"analysis_type": "jury"}
        assert not Path(path + ".roi.json").exists()
        assert settings.scenario_model_path != settings.model_path
        assert 0 < timeout_sec <= 295
        return projected_result(positive)

    monkeypatch.setattr(api, "_run_child", worker)
    with TestClient(api.create_app(configured)) as client:
        accepted = submit(client)
        assert accepted.status_code == 202
        result = terminal(client, accepted.json()["job_id"])
        assert result["status"] == "done"
        assert result["verdict"] == ("violation" if positive else "no_violation")
        assert result["violations"] == projected_result(positive)["violations"]
        assert result["internal_verdict"] == ("review_required" if positive else "no_candidate")
        assert result["provenance"] == projected_result()["provenance"]
        assert result["coverage"] == projected_result()["coverage"]
        assert result["video_duration_sec"] == 1.0
        assert result["rules_status"] == "unverified"
        assert result["demo"] is False
        with api._connect(configured) as db:
            assert db.execute("SELECT analysis_type FROM jobs").fetchone()[0] == "jury"
    assert len(seen) == 1
    assert not list(configured.upload_dir.glob("*.mov*"))


def test_jury_readiness_is_explicit_without_claiming_weights_loaded(tmp_path):
    configured = configuration(tmp_path)
    with TestClient(api.create_app(configured)) as client:
        data = client.get("/api/v1/readiness", headers=HEADERS).json()
    assert data["configured_for_jury_contract"] is True
    assert data["omitted_analysis_type_routes_to"] == "bank_scenarios"
    assert data["actual_analyzer"] == "src.vision.scenario_classifier:analyze_video"
    assert data["model_load_verified"] is False
    assert data["jury_contract_compatible"] is None
    assert data["analysis_work_budget_sec"] + data["shutdown_reserve_sec"] == 300


@pytest.mark.parametrize("overrides", [{"analyzer_mode": "demo"}, {"default_analysis_type": None},
                                      {"jury_analyzer": "safe_state"}])
def test_jury_profile_rejects_incompatible_configuration(tmp_path, overrides):
    with pytest.raises(ValueError, match="JURY_ANALYZER"):
        configuration(tmp_path, **overrides)


def test_jury_setting_can_be_selected_from_environment(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("JURY_ANALYZER", "bank_scenarios")
    monkeypatch.setenv("DEFAULT_ANALYSIS_TYPE", "jury")
    monkeypatch.setenv("ANALYZER_MODE", "real")
    assert get_settings().jury_analyzer == "bank_scenarios"


@pytest.mark.parametrize("missing", ["model.npz", "metadata.json", "backbone.onnx", None])
def test_missing_scenario_weights_cannot_use_safe_weights_or_demo(tmp_path, monkeypatch, missing):
    configured = configuration(tmp_path, **({"scenario_model_path": None} if missing is None else {}))
    if missing:
        (Path(configured.scenario_model_path) / missing).unlink()
    monkeypatch.setattr(api, "_run_child", lambda *args: pytest.fail("Worker must not start"))
    with TestClient(api.create_app(configured)) as client:
        response = submit(client)
        assert response.status_code == 503
        assert "missing" in response.json()["detail"]
        assert str(tmp_path) not in response.text
        assert "verdict" not in response.json()
    assert not list(configured.upload_dir.glob("*.mov*"))


def test_wrong_keys_rejected_on_both_routes_and_unknown_job_is_404(tmp_path):
    with TestClient(api.create_app(configuration(tmp_path))) as client:
        for headers in ({}, {"X-API-Key": "incorrect"}):
            assert client.post("/api/v1/analyze", headers=headers,
                               files={"file": ("fixture.mov", b"video")}).status_code == 401
            assert client.get("/api/v1/jobs/unknown", headers=headers).status_code == 401
        assert client.get("/api/v1/jobs/unknown", headers=HEADERS).status_code == 404


def test_explicit_safe_mode_still_requires_roi_and_uses_its_own_route(tmp_path, monkeypatch):
    configured = configuration(tmp_path)
    def worker(path, settings, timeout_sec, stop_event=None):
        assert json.loads(Path(path + ".analysis.json").read_text()) == {"analysis_type": "safe_state"}
        assert json.loads(Path(path + ".api-mode.json").read_text()) == {"analysis_type": "safe_state"}
        assert json.loads(Path(path + ".roi.json").read_text()) == {"roi": [0., 0., 1., 1.]}
        return {"verdict": "no_event", "violations": [], "analysis_scope": "safe_state", "demo": False}
    monkeypatch.setattr(api, "_run_child", worker)
    with TestClient(api.create_app(configured)) as client:
        assert submit(client, analysis_type="safe_state").status_code == 422
        accepted = submit(client, analysis_type="safe_state", roi="[0,0,1,1]")
        assert accepted.status_code == 202
        result = terminal(client, accepted.json()["job_id"])
        assert result["verdict"] == "no_event"
        assert result["analysis_scope"] == "safe_state"


@pytest.mark.parametrize("failure", ["exception", "scoped", "demo", "nan", "wrong_analyzer"])
def test_worker_failure_or_invalid_projection_never_becomes_negative(tmp_path, monkeypatch, failure):
    configured = configuration(tmp_path)
    def worker(path, settings, timeout_sec, stop_event=None):
        if failure == "exception":
            raise RuntimeError("private-path-and-secret")
        value = projected_result()
        if failure == "scoped": value["verdict"] = "no_candidate"
        if failure == "demo": value["demo"] = True
        if failure == "nan": value["coverage"] = {"invalid": float("nan")}
        if failure == "wrong_analyzer": value["provenance"]["actual_analyzer"] = "safe_state"
        return value
    monkeypatch.setattr(api, "_run_child", worker)
    with TestClient(api.create_app(configured)) as client:
        accepted = submit(client)
        result = terminal(client, accepted.json()["job_id"])
        assert result["status"] == "failed"
        assert "verdict" not in result
        assert "private" not in result["error"]
    assert not list(configured.upload_dir.glob("*.mov*"))


def test_parallel_jobs_keep_independent_modes_and_results(tmp_path, monkeypatch):
    configured = configuration(tmp_path)
    seen = {}
    def worker(path, settings, timeout_sec, stop_event=None):
        mode = json.loads(Path(path + ".api-mode.json").read_text())["analysis_type"]
        selection = json.loads(Path(path + ".analysis.json").read_text())["analysis_type"]
        seen[Path(path).stem] = (mode, selection)
        if mode == "jury": return projected_result(True)
        return {"verdict": "no_event" if mode == "safe_state" else "no_candidate",
                "violations": [], "analysis_scope": mode, "demo": False}
    monkeypatch.setattr(api, "_run_child", worker)
    with TestClient(api.create_app(configured)) as client:
        requests = [{}, {"analysis_type": "safe_state", "roi": "[0,0,1,1]"},
                    {"analysis_type": "bank_scenarios"}]
        with ThreadPoolExecutor(max_workers=3) as pool:
            accepted = list(pool.map(lambda fields: submit(client, **fields), requests))
        assert all(response.status_code == 202 for response in accepted)
        ids = [response.json()["job_id"] for response in accepted]
        assert len(set(ids)) == 3
        results = [terminal(client, job) for job in ids]
        assert [result["verdict"] for result in results] == ["violation", "no_event", "no_candidate"]
        assert [seen[job] for job in ids] == [("jury", "bank_scenarios"),
                                            ("safe_state", "safe_state"), ("bank_scenarios", "bank_scenarios")]
    assert not list(configured.upload_dir.glob("*.mov*"))


def test_late_result_is_failed_and_uploads_are_removed(tmp_path, monkeypatch):
    configured = configuration(tmp_path, analyzer_timeout_sec=1)
    def worker(path, settings, timeout_sec, stop_event=None):
        assert timeout_sec < 1
        time.sleep(1.05)
        return projected_result()
    monkeypatch.setattr(api, "_run_child", worker)
    with TestClient(api.create_app(configured)) as client:
        result = terminal(client, submit(client).json()["job_id"])
        assert result["status"] == "failed"
        assert result["error"] == "Analysis timed out"
        assert "verdict" not in result
    assert not list(configured.upload_dir.glob("*.mov*"))


def test_queue_wait_consumes_the_same_request_deadline(tmp_path, monkeypatch):
    configured = configuration(tmp_path, analyzer_timeout_sec=3)
    budgets = []
    def worker(path, settings, timeout_sec, stop_event=None):
        budgets.append(timeout_sec)
        if len(budgets) == 1:
            time.sleep(.2)
        return projected_result()
    monkeypatch.setattr(api, "_run_child", worker)
    with TestClient(api.create_app(configured)) as client:
        first = submit(client)
        second = submit(client)
        assert first.status_code == second.status_code == 202
        assert terminal(client, first.json()["job_id"])["status"] == "done"
        assert terminal(client, second.json()["job_id"])["status"] == "done"
    assert budgets[1] < budgets[0] - .1


def test_job_mode_sidecar_cannot_change_the_persisted_contract(tmp_path, monkeypatch):
    configured = configuration(tmp_path)
    def worker(path, settings, timeout_sec, stop_event=None):
        Path(path + ".api-mode.json").write_text('{"analysis_type":"bank_scenarios"}')
        return {"verdict": "no_candidate", "violations": [], "analysis_scope": "bank_scenarios", "demo": False}
    monkeypatch.setattr(api, "_run_child", worker)
    with TestClient(api.create_app(configured)) as client:
        result = terminal(client, submit(client).json()["job_id"])
        assert result["status"] == "failed"
        assert "verdict" not in result
    assert not list(configured.upload_dir.glob("*.mov*"))


def test_expired_results_are_not_returned_and_wrong_saved_contract_fails(tmp_path):
    configured = configuration(tmp_path, job_ttl_sec=10)
    with TestClient(api.create_app(configured)) as client:
        with api._connect(configured) as db:
            for job, updated, verdict in (("expired", time.time() - 20, "no_violation"),
                                          ("wrong-mode", time.time(), "no_candidate")):
                db.execute("INSERT INTO jobs(job_id,status,video_path,created_at,updated_at,result_json,analysis_type) VALUES(?,?,?,?,?,?,?)",
                           (job, "done", str(configured.upload_dir / (job + ".mov")), updated, updated,
                            json.dumps({"verdict": verdict, "violations": []}), "jury"))
        assert client.get("/api/v1/jobs/expired", headers=HEADERS).status_code == 404
        result = client.get("/api/v1/jobs/wrong-mode", headers=HEADERS).json()
        assert result["status"] == "failed"
        assert "verdict" not in result


def test_existing_database_migration_preserves_completed_legacy_job(tmp_path):
    configured = configuration(tmp_path)
    with sqlite3.connect(configured.database_path) as db:
        db.execute("CREATE TABLE jobs(job_id TEXT PRIMARY KEY,status TEXT NOT NULL,video_path TEXT NOT NULL,created_at REAL NOT NULL,updated_at REAL NOT NULL,result_json TEXT,error TEXT,demo INTEGER NOT NULL DEFAULT 0)")
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)", ("legacy", "done", "old.mov", time.time(),
                   time.time(), '{"verdict":"no_event","violations":[],"analysis_scope":"safe_state"}', None, 0))
    with TestClient(api.create_app(configured)) as client:
        result = client.get("/api/v1/jobs/legacy", headers=HEADERS).json()
        assert result["status"] == "done"
        assert result["verdict"] == "no_event"


def test_slow_request_body_has_a_deadline_before_multipart_and_queue():
    called, sent = [], []
    async def downstream(scope, receive, send): called.append(True)
    async def receive():
        await asyncio.sleep(.1)
        return {"type": "http.request", "body": b"body", "more_body": False}
    async def send(message): sent.append(message)
    scope = {"type": "http", "method": "POST", "path": "/api/v1/analyze",
             "headers": [(b"x-api-key", b"jury-test-key")]}
    guard = api.ApiRequestGuard(downstream, "jury-test-key", 1024, timeout_sec=.02)
    asyncio.run(guard(scope, receive, send))
    assert called == []
    assert sent[0]["status"] == 408
    assert "analysis_request_started_at" in scope


@pytest.mark.parametrize("condition", ["partial_payload", "producer_already_dead", "slow_process_start"])
def test_transport_cannot_return_a_result_after_the_deadline(tmp_path, monkeypatch, condition):
    """Simulate a readable pipe whose complete framed payload is still pending."""
    release, entered = threading.Event(), threading.Event()

    class Receiver:
        closed = False
        def poll(self, timeout=0): return True  # A partial header is available.
        def recv(self):
            entered.set()
            release.wait(2)
            return True, projected_result()
        def close(self):
            self.closed = True
            release.set()

    class Child:
        alive = False
        terminated = False
        def start(self):
            self.alive = condition != "producer_already_dead"
            if condition == "slow_process_start": time.sleep(.08)
        def is_alive(self): return self.alive
        def terminate(self):
            self.terminated = True
            self.alive = False
            release.set()
        def kill(self): self.terminate()
        def join(self, timeout=0): pass

    receiver, child = Receiver(), Child()
    context = SimpleNamespace(Pipe=lambda **kwargs: (receiver, SimpleNamespace(close=lambda: None)),
                              Process=lambda **kwargs: child)
    monkeypatch.setattr(api.multiprocessing, "get_context", lambda method: context)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        api._run_child("synthetic.mov", configuration(tmp_path), .03)
    assert time.monotonic() - started < .5
    assert entered.is_set()
    assert receiver.closed
    assert not child.alive
    if condition != "producer_already_dead":
        assert child.terminated
