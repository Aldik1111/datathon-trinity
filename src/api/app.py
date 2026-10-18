"""FastAPI service for queued video analysis."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import multiprocessing
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from .settings import Settings, get_settings
from src.vision.scenario_contracts import ANALYSIS_CALLABLES

logger = logging.getLogger(__name__)
_CHUNK_SIZE = 1024 * 1024
_EXPERIMENTAL_ANALYZERS = {
    "src.vision.safe_classifier:analyze_video": "safe_state",
    "src.vision.scenario_classifier:analyze_video": "bank_scenarios",
}


class ApiRequestGuard:
    """Authenticate before parsing multipart bodies and cap the full request."""

    def __init__(self, app, api_key: str, max_body_bytes: int):
        self.app = app
        self.api_key = api_key
        # Allow modest multipart headers/boundaries around the uploaded file.
        self.max_body_bytes = max_body_bytes + 64 * 1024

    async def _reject(self, scope, receive, send, status: int, message: str) -> None:
        response = JSONResponse(status_code=status, content={"detail": message})
        await response(scope, receive, send)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/v1/"):
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        supplied = headers.get(b"x-api-key", b"")
        if not hmac.compare_digest(supplied, self.api_key.encode("utf-8")):
            await self._reject(scope, receive, send, 401, "Invalid or missing API key")
            return
        if scope.get("method") != "POST" or scope.get("path") != "/api/v1/analyze":
            await self.app(scope, receive, send)
            return
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                if int(raw_length) > self.max_body_bytes:
                    await self._reject(scope, receive, send, 413, "Upload exceeds size limit")
                    return
            except ValueError:
                await self._reject(scope, receive, send, 400, "Invalid content length")
                return
        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as body:
            total = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self.max_body_bytes:
                    await self._reject(scope, receive, send, 413, "Upload exceeds size limit")
                    return
                body.write(chunk)
                if not message.get("more_body", False):
                    break
            body.seek(0)

            async def replay_receive():
                chunk = body.read(64 * 1024)
                return {"type": "http.request", "body": chunk, "more_body": bool(chunk)}

            await self.app(scope, replay_receive, send)


class SafeAnalysisFailure(RuntimeError):
    """A fixed public message from the adapter that contains no raw exception data."""


def _remove_upload(path: str | Path) -> None:
    """Remove one job's temporary movie, calibration, and analyzer selection."""
    for target in (Path(path), Path(str(path) + ".roi.json"), Path(str(path) + ".analysis.json"),
                   Path(str(path) + ".api-mode.json")):
        with suppress(OSError):
            target.unlink()


def _parse_roi(value: str | None, required: bool) -> list[float] | None:
    if value is None and not required:
        return None
    try:
        roi = json.loads(value) if value is not None and len(value) < 256 else None
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError
        if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in roi):
            raise ValueError
        x, y, w, h = roi
        if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x + w <= 1 and y + h <= 1):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(status_code=422, detail="Select a valid safe region: [x, y, width, height] within 0..1") from None
    return [float(n) for n in roi]


@contextmanager
def _connect(settings: Settings):
    connection = sqlite3.connect(settings.database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize(settings: Settings) -> None:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(settings) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                video_path TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                result_json TEXT,
                error TEXT,
                demo INTEGER NOT NULL DEFAULT 0
            )"""
        )
        # A worker is deliberately single-process. If the server stopped while
        # analyzing, the job cannot safely be resumed from an unknown model state.
        interrupted = db.execute(
            "SELECT video_path FROM jobs WHERE status='processing'"
        ).fetchall()
        db.execute(
            "UPDATE jobs SET status='failed', error='Analysis interrupted by server restart', updated_at=? WHERE status='processing'",
            (time.time(),),
        )
    for row in interrupted:
        _remove_upload(row["video_path"])


def _cleanup_expired(settings: Settings) -> None:
    cutoff = time.time() - settings.job_ttl_sec
    with _connect(settings) as db:
        rows = db.execute(
            "SELECT video_path FROM jobs WHERE status!='processing' AND updated_at < ?", (cutoff,)
        ).fetchall()
        db.execute("DELETE FROM jobs WHERE status!='processing' AND updated_at < ?", (cutoff,))
    for row in rows:
        _remove_upload(row["video_path"])


def _child_analyze(send_conn: Any, video_path: str, settings: Settings) -> None:
    """Run analysis in an isolated child so timeout termination is reliable."""
    try:
        # Adapter import is delayed so API startup and health checks remain usable
        # when an optional model dependency is unavailable.
        from src.vision.adapter import AnalysisError, run_analysis

        try:
            result = run_analysis(video_path, settings)
        except AnalysisError as exc:
            # The adapter uses fixed messages for this public exception type.
            send_conn.send((False, {"code": "analysis_failed", "message": str(exc)[:200]}))
            return
        send_conn.send((True, result))
    except BaseException as exc:
        # Unexpected exception text can contain model paths, tokens, or input data.
        send_conn.send((False, {"code": "analysis_failed", "message": "Analysis failed"}))
    finally:
        send_conn.close()


def _run_child(
    video_path: str,
    settings: Settings,
    timeout_sec: float,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    ctx = multiprocessing.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child_analyze, args=(send_conn, video_path, settings))
    deadline = time.monotonic() + max(0.0, timeout_sec)
    process.start()
    send_conn.close()
    try:
        while time.monotonic() < deadline:
            if recv_conn.poll(0.1):
                if time.monotonic() >= deadline:
                    process.terminate()
                    process.join(timeout=3)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=3)
                    raise TimeoutError("Analysis exceeded the job deadline")
                ok, payload = recv_conn.recv()
                process.join(timeout=1)
                if not ok:
                    raise SafeAnalysisFailure(payload.get("message", "Analysis failed"))
                if not isinstance(payload, dict):
                    raise RuntimeError("Analyzer returned an invalid result")
                return payload
            if not process.is_alive():
                process.join()
                if recv_conn.poll():
                    ok, payload = recv_conn.recv()
                    if ok and isinstance(payload, dict):
                        return payload
                    raise SafeAnalysisFailure(
                        payload.get("message", "Analysis failed")
                        if isinstance(payload, dict)
                        else "Analysis failed"
                    )
                raise RuntimeError("Analyzer exited without a result")
            if stop_event is not None and stop_event.is_set():
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
                raise RuntimeError("Analysis stopped")
        process.terminate()
        process.join(timeout=3)
        if process.is_alive():
            process.kill()
            process.join(timeout=3)
        raise TimeoutError("Analysis exceeded the 300 second job deadline")
    finally:
        recv_conn.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)


def _public_result(job_id: str, result_json: str | None, demo: int) -> dict[str, Any]:
    payload = json.loads(result_json or "{}")
    verdict = payload.get("verdict")
    if verdict not in {"violation", "no_violation", "review_required", "no_event", "no_candidate"}:
        raise RuntimeError("Analyzer returned an invalid verdict")
    raw_violations = payload.get("violations", [])
    if not isinstance(raw_violations, list):
        raise RuntimeError("Analyzer returned invalid violations")
    violations = []
    for item in raw_violations:
        if not isinstance(item, dict):
            continue
        frames = item.get("evidence_frames", [])
        violations.append(
            {
                "type": str(item.get("type", "unknown")),
                "timecode_sec": float(item.get("timecode_sec", 0)),
                "description": str(item.get("description", "")),
                "evidence_frames": [
                    {
                        "timestamp_sec": float(frame.get("timestamp_sec", 0)),
                        "image_base64": str(frame.get("image_base64", "")),
                    }
                    for frame in frames
                    if isinstance(frame, dict)
                ],
                **({"rule": item["rule"]} if "rule" in item else {}),
                **({"rule_status": item["rule_status"]} if "rule_status" in item else {}),
                **{key: item[key] for key in ("start_sec", "end_sec", "confidence") if key in item},
            }
        )
    response = {
        "job_id": job_id,
        "status": "done",
        "verdict": verdict,
        "violations": violations,
        "demo": bool(demo),
    }
    # Ruleset metadata is safe to expose and useful to clients when supplied.
    for key in ("ruleset_version", "ruleset_hash", "model_version", "analysis_scope", "summary", "observations",
                "feature_version", "score_semantics", "class_results", "limitations", "model_support_policy"):
        if key in payload:
            response[key] = payload[key]
    for key in ("rule", "rule_metadata", "ruleset"):
        if key in payload:
            response[key] = payload[key]
    return response


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or get_settings()
    state: dict[str, Any] = {
        "queue": None,
        "worker": None,
        "cleanup": None,
        "stop_event": None,
        "admission_lock": None,
        "reservations": 0,
    }

    async def worker_loop(queue: asyncio.Queue[str]) -> None:
        while True:
            job_id = await queue.get()
            try:
                with _connect(configured) as db:
                    row = db.execute(
                        "SELECT * FROM jobs WHERE job_id=?", (job_id,)
                    ).fetchone()
                    if row is None or row["status"] != "processing":
                        continue
                    remaining = configured.analyzer_timeout_sec - (
                        time.time() - row["created_at"]
                    )
                    if remaining <= 0:
                        raise TimeoutError("Analysis exceeded the 300 second job deadline")
                    video_path = row["video_path"]
                analysis_task = asyncio.create_task(
                    asyncio.to_thread(
                        _run_child, video_path, configured, remaining, state["stop_event"]
                    )
                )
                try:
                    result = await asyncio.shield(analysis_task)
                except asyncio.CancelledError:
                    with suppress(Exception):
                        await asyncio.shield(analysis_task)
                    raise
                try:
                    api_mode = json.loads(
                        Path(video_path + ".api-mode.json").read_text(encoding="utf-8")
                    )["analysis_type"]
                except (OSError, ValueError, KeyError, TypeError):
                    raise SafeAnalysisFailure("Analysis mode metadata is unavailable") from None
                if api_mode == "jury" and (
                    result.get("analysis_scope") is not None
                    or result.get("verdict") not in {"violation", "no_violation"}
                ):
                    raise SafeAnalysisFailure(
                        "Configured analyzer is not jury-compatible; select an explicit experimental mode"
                    )
                demo = bool(result.get("demo", configured.analyzer_mode == "demo"))
                encoded = json.dumps(result, ensure_ascii=False)
                with _connect(configured) as db:
                    db.execute(
                        "UPDATE jobs SET status='done', result_json=?, demo=?, error=NULL, updated_at=? WHERE job_id=? AND status='processing'",
                        (encoded, int(demo), time.time(), job_id),
                    )
                _remove_upload(video_path)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Never return model exception text, which can contain paths or secrets.
                safe_message = (
                    "Analysis timed out"
                    if isinstance(exc, TimeoutError)
                    else str(exc)[:200] if isinstance(exc, SafeAnalysisFailure)
                    else "Analysis stopped" if str(exc) == "Analysis stopped" else "Analysis failed"
                )
                logger.warning("Analysis failed for job %s (%s)", job_id, type(exc).__name__)
                with _connect(configured) as db:
                    row = db.execute(
                        "SELECT video_path FROM jobs WHERE job_id=?", (job_id,)
                    ).fetchone()
                    db.execute(
                        "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE job_id=? AND status='processing'",
                        (safe_message, time.time(), job_id),
                    )
                if row:
                    _remove_upload(row["video_path"])
            finally:
                queue.task_done()

    async def cleanup_loop() -> None:
        interval = min(60, max(1, configured.job_ttl_sec // 4))
        while True:
            await asyncio.sleep(interval)
            await asyncio.to_thread(_cleanup_expired, configured)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _initialize(configured)
        _cleanup_expired(configured)
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=configured.max_pending_jobs)
        state["queue"] = queue
        state["stop_event"] = threading.Event()
        state["admission_lock"] = asyncio.Lock()
        state["worker"] = asyncio.create_task(worker_loop(queue))
        state["cleanup"] = asyncio.create_task(cleanup_loop())
        app.state.settings = configured
        app.state.job_queue = queue
        yield
        state["stop_event"].set()
        state["worker"].cancel()
        with suppress(asyncio.CancelledError):
            await state["worker"]
        state["cleanup"].cancel()
        with suppress(asyncio.CancelledError):
            await state["cleanup"]
        with _connect(configured) as db:
            paths = db.execute(
                "SELECT video_path FROM jobs WHERE status='processing'"
            ).fetchall()
            db.execute(
                "UPDATE jobs SET status='failed', error='Analysis interrupted by server shutdown', updated_at=? WHERE status='processing'",
                (time.time(),),
            )
        for row in paths:
            _remove_upload(row["video_path"])

    app = FastAPI(title="Trinity Video Analysis API", version="1.0.0", lifespan=lifespan)
    app.state.settings = configured
    app.add_middleware(
        ApiRequestGuard,
        api_key=configured.api_key,
        max_body_bytes=configured.max_upload_mb * 1024 * 1024,
    )

    def require_key(x_api_key: str | None) -> None:
        if x_api_key is None or not hmac.compare_digest(
            x_api_key.encode("utf-8"), configured.api_key.encode("utf-8")
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.post("/api/v1/analyze", status_code=202)
    async def analyze_video(
        request: Request,
        file: UploadFile = File(...), x_api_key: str | None = Header(default=None),
        roi: str | None = Form(default=None),
        analysis_type: str | None = Form(default=None),
    ) -> JSONResponse:
        require_key(x_api_key)
        # FastAPI converts an empty optional form string to its default. Inspect
        # the cached form too, so an explicitly empty/duplicate selection is not
        # silently treated as the omitted field used by older clients.
        selections = (await request.form()).getlist("analysis_type")
        if selections and (len(selections) != 1 or analysis_type not in ANALYSIS_CALLABLES):
            raise HTTPException(status_code=422, detail="Select safe_state or bank_scenarios")
        inferred_default = (_EXPERIMENTAL_ANALYZERS.get(configured.analyzer_callable, "jury")
                            if configured.analyzer_mode == "real" else "jury")
        selected_type = analysis_type or configured.default_analysis_type or inferred_default
        explicit_route = analysis_type is not None or configured.default_analysis_type is not None
        if selected_type != "jury" and configured.analyzer_mode != "real":
            raise HTTPException(status_code=422, detail="Analyzer selection requires real mode")
        if (selected_type == "jury" and configured.analyzer_mode == "real"
                and configured.analyzer_callable in _EXPERIMENTAL_ANALYZERS):
            raise HTTPException(
                status_code=409,
                detail="The configured analyzer is experimental and cannot serve the jury verdict contract; choose an explicit experimental mode or configure a jury-compatible analyzer",
            )
        needs_roi = selected_type == "safe_state" or (
            selected_type == "jury" and configured.analyzer_mode == "real"
            and configured.analyzer_callable == ANALYSIS_CALLABLES["safe_state"]
        )
        selected_roi = _parse_roi(roi, needs_roi)
        _cleanup_expired(configured)
        if not file.filename or Path(file.filename).suffix.lower() != ".mov":
            raise HTTPException(status_code=415, detail="Only .mov files are accepted")
        queue: asyncio.Queue[str] = state["queue"]
        if queue is None:
            raise HTTPException(status_code=503, detail="Analysis queue is full")
        lock: asyncio.Lock = state["admission_lock"]
        async with lock:
            with _connect(configured) as db:
                active = db.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status='processing'"
                ).fetchone()[0]
            if active + state["reservations"] >= configured.max_pending_jobs:
                raise HTTPException(status_code=503, detail="Analysis queue is full")
            state["reservations"] += 1
        reserved = True

        job_id = str(uuid.uuid4())
        target = configured.upload_dir / f"{job_id}.mov"
        size = 0
        try:
            with target.open("xb") as output:
                while chunk := await file.read(_CHUNK_SIZE):
                    size += len(chunk)
                    if size > configured.max_upload_mb * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="Upload exceeds size limit")
                    output.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            if selected_roi is not None:
                Path(str(target) + ".roi.json").write_text(json.dumps({"roi": selected_roi}), encoding="utf-8")
            if selected_type != "jury" and explicit_route:
                Path(str(target) + ".analysis.json").write_text(
                    json.dumps({"analysis_type": selected_type}), encoding="utf-8"
                )
            Path(str(target) + ".api-mode.json").write_text(
                json.dumps({"analysis_type": selected_type}), encoding="utf-8"
            )
        except BaseException:
            _remove_upload(target)
            async with lock:
                state["reservations"] -= 1
                reserved = False
            raise
        finally:
            await file.close()
        try:
            async with lock:
                now = time.time()
                with _connect(configured) as db:
                    db.execute(
                        "INSERT INTO jobs(job_id,status,video_path,created_at,updated_at,demo) VALUES(?,?,?,?,?,?)",
                        (
                            job_id,
                            "processing",
                            str(target),
                            now,
                            now,
                            int(configured.analyzer_mode == "demo"),
                        ),
                    )
                try:
                    queue.put_nowait(job_id)
                except asyncio.QueueFull:
                    with _connect(configured) as db:
                        db.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
                    raise HTTPException(status_code=503, detail="Analysis queue is full")
                state["reservations"] -= 1
                reserved = False
        except BaseException:
            _remove_upload(target)
            raise
        finally:
            if reserved:
                async with lock:
                    state["reservations"] -= 1
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "status": "processing", "demo": configured.analyzer_mode == "demo"},
        )

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        require_key(x_api_key)
        _cleanup_expired(configured)
        with _connect(configured) as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if row["status"] == "done":
            return _public_result(row["job_id"], row["result_json"], row["demo"])
        response: dict[str, Any] = {
            "job_id": row["job_id"],
            "status": row["status"],
            "demo": bool(row["demo"]) if row["status"] == "failed" else configured.analyzer_mode == "demo",
        }
        if row["status"] == "failed":
            response["error"] = row["error"] or "Analysis failed"
        return response

    @app.get("/api/v1/readiness")
    async def readiness(x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        """Report routing and declared contract readiness without loading a model."""
        require_key(x_api_key)
        configured_experimental = _EXPERIMENTAL_ANALYZERS.get(configured.analyzer_callable)
        default_mode = configured.default_analysis_type or (configured_experimental if configured.analyzer_mode == "real" else None) or "jury"
        effective_mode = (
            "rejected_experimental_default"
            if configured.analyzer_mode == "real" and configured.default_analysis_type == "jury" and configured_experimental is not None
            else default_mode
        )
        configured_route = configured_experimental if effective_mode == "rejected_experimental_default" else effective_mode
        configured_for_jury = effective_mode == "jury"
        compatible = (
            False
            if (configured.analyzer_mode == "demo" or configured_experimental is not None
                or effective_mode in {"safe_state", "bank_scenarios", "rejected_experimental_default"})
            else None
        )
        model_configured = (
            bool(configured.model_path)
            if configured_route == "jury"
            else bool(configured.model_path)
            if configured_route == "safe_state" or configured.default_analysis_type is None
            else bool(configured.scenario_model_path)
        )
        reason = (
            "The configured analyzer uses an experimental scoped verdict contract."
            if configured_experimental is not None and effective_mode == "rejected_experimental_default"
            else "Demo mode returns a synthetic result, not a jury analysis."
            if configured.analyzer_mode == "demo"
            else "Jury contract compatibility is unknown until a real analyzer output passes runtime validation."
            if configured_for_jury
            else "The default route is an explicitly scoped experimental analysis."
        )
        return {
            "api_configured": bool(configured.api_key),
            "model_configured": model_configured,
            "model_load_verified": False,
            "analyzer_mode": configured.analyzer_mode,
            "default_analysis_type": effective_mode,
            "omitted_analysis_type_routes_to": effective_mode,
            "configured_for_jury_contract": configured_for_jury,
            "jury_contract_compatible": compatible,
            "jury_contract_basis": reason,
            "analysis_timeout_sec": configured.analyzer_timeout_sec,
            "maximum_analysis_deadline_sec": 300,
            "experimental_modes": ["safe_state", "bank_scenarios"],
        }

    return app
