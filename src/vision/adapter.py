"""Validate local model output and adapt it to the jury's JSON contract."""
from __future__ import annotations

import base64
import importlib
import json
import math
import os
from io import BytesIO
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

from .contracts import AnalysisResult, Event, EvidenceFrame


class AnalysisError(ValueError):
    """Only fixed, non-sensitive messages should be raised here."""


def _number(value: Any, duration: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError("Invalid event or frame timestamp")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= duration + 0.05:
        raise AnalysisError("Event or frame timestamp is outside the video")
    return value


def _mapping(value: Any) -> dict:
    if is_dataclass(value):
        return asdict(value)
    if not isinstance(value, dict):
        raise AnalysisError("Model result does not match the integration contract")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise AnalysisError("Model returned an invalid event description or type")
    return value


def load_rules(path: str | None) -> dict:
    if not path:
        return {}
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise AnalysisError("Rules registry is missing or invalid") from None
    if not isinstance(records, list):
        raise AnalysisError("Rules registry must contain a list")
    rules = {}
    for rule in records:
        if not isinstance(rule, dict) or rule.get("verified") is not True:
            continue
        required = ("rule_id", "title", "url", "section", "checked_at", "limitations")
        if any(not isinstance(rule.get(key), str) or not rule[key].strip() for key in required):
            raise AnalysisError("Verified rule is incomplete")
        parsed = urlparse(rule["url"])
        if parsed.scheme != "https" or not parsed.netloc:
            raise AnalysisError("Verified rule requires an HTTPS source URL")
        if rule["rule_id"] in rules:
            raise AnalysisError("Duplicate rule identifier")
        rules[rule["rule_id"]] = {key: rule[key] for key in required}
    return rules


def normalize_result(raw: Any, duration: float, rules: dict) -> dict:
    result = _mapping(raw)
    verdict, events = result.get("verdict"), result.get("events")
    if verdict not in {"violation", "no_violation"} or not isinstance(events, list):
        raise AnalysisError("Model result does not match the integration contract")
    if (verdict == "violation") != bool(events) or len(events) > 100:
        raise AnalysisError("Model verdict is inconsistent with its events")
    violations = []
    evidence_size = 0
    for raw_event in events:
        event = _mapping(raw_event)
        frames = event.get("evidence_frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
            raise AnalysisError("Each event requires 1 to 3 JPEG evidence frames")
        normalized_frames = []
        for raw_frame in frames:
            frame = _mapping(raw_frame)
            jpeg = frame.get("jpeg_bytes")
            if not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= 4 * 1024 * 1024:
                raise AnalysisError("Evidence must contain JPEG bytes (at most 4 MiB)")
            if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                raise AnalysisError("Evidence is not a JPEG image")
            evidence_size += len(jpeg)
            if evidence_size > 12 * 1024 * 1024:
                raise AnalysisError('Total evidence exceeds 12 MiB')
            try:
                with Image.open(BytesIO(jpeg)) as image:
                    if image.format != 'JPEG' or image.width * image.height > 16_000_000:
                        raise AnalysisError('Evidence JPEG dimensions exceed the limit')
            except AnalysisError:
                raise
            except (OSError, ValueError, Image.DecompressionBombError):
                raise AnalysisError('Evidence JPEG cannot be decoded') from None
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise AnalysisError("Evidence JPEG cannot be decoded")
            normalized_frames.append({
                "timestamp_sec": _number(frame.get("timestamp_sec"), duration),
                "image_base64": base64.b64encode(jpeg).decode("ascii"),
            })
        violation = {
            "type": _text(event.get("event_type")),
            "timecode_sec": _number(event.get("timecode_sec"), duration),
            "description": _text(event.get("description")),
            "evidence_frames": normalized_frames,
        }
        rule_id = event.get("rule_id")
        if rule_id is not None and not isinstance(rule_id, str):
            raise AnalysisError("Invalid rule identifier")
        violation["rule"] = rules.get(rule_id)
        violation["rule_status"] = "verified" if rule_id in rules else "unverified"
        violations.append(violation)
    return {"verdict": verdict, "violations": violations}


def inspect_video(path: str) -> tuple[float, bytes]:
    """Decode the whole file locally; use decoder timestamps where available.

    OpenCV may silently accept some damaged containers. Full decode and reported
    frame count checks are useful validation, not a forensic integrity guarantee.
    """
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise AnalysisError("Video cannot be opened")
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        count, max_timestamp, first_jpeg = 0, 0.0, None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if math.isfinite(timestamp) and timestamp >= 0:
                max_timestamp = max(max_timestamp, timestamp)
            if first_jpeg is None:
                height, width = frame.shape[:2]
                if max(height, width) > 1280:
                    frame = cv2.resize(frame, (round(width * 1280 / max(height, width)), round(height * 1280 / max(height, width))))
                encoded, image = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not encoded:
                    raise AnalysisError("Video frame cannot be encoded")
                first_jpeg = image.tobytes()
        if count == 0 or first_jpeg is None:
            raise AnalysisError("Video contains no decodable frames")
        if math.isfinite(expected) and expected > 0 and count + 1 < expected:
            raise AnalysisError("Video ended before its declared frame count")
        interval = 1 / fps if math.isfinite(fps) and fps > 0 else 0
        duration = max_timestamp + interval if max_timestamp > 0 else count * interval
        if not math.isfinite(duration) or duration <= 0:
            raise AnalysisError("Video has no usable timing information")
        return duration, first_jpeg
    finally:
        capture.release()


def run_analysis(video_path: str, settings: Any) -> dict:
    duration, first_jpeg = inspect_video(str(video_path))
    if settings.analyzer_mode == "demo":
        # This deliberately synthetic event tests plumbing, not recognition.
        raw = AnalysisResult("violation", [Event(
            event_type="ТЕСТ: демонстрация карточки события",
            timecode_sec=0.0,
            description="Тестовый ответ для проверки интерфейса. Содержимое видео не классифицировалось; это не результат обнаружения нарушения.",
            evidence_frames=[EvidenceFrame(0.0, first_jpeg)],
        )])
        rules = {}
    else:
        # The model's stable public signature remains analyze_video(video_path).
        # Its optional local weight location is supplied through the environment.
        if settings.model_path:
            os.environ["MODEL_PATH"] = str(settings.model_path)
        try:
            module_name, function_name = settings.analyzer_callable.split(":", 1)
            analyzer = getattr(importlib.import_module(module_name), function_name)
        except (ValueError, ImportError, AttributeError):
            raise AnalysisError("Local model is not installed. Configure ANALYZER_CALLABLE and MODEL_PATH") from None
        try:
            raw = analyzer(str(video_path))
        except Exception:
            raise AnalysisError("Local model failed to analyze the video") from None
        rules = load_rules(settings.rules_path)
    result = normalize_result(raw, duration, rules)
    result["demo"] = settings.analyzer_mode == "demo"
    return result
