"""Validate local model output and adapt it to the jury's JSON contract."""
from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from .contracts import AnalysisResult, Event, EvidenceFrame
from .scenario_contracts import ANALYSIS_CALLABLES, SCENARIO_TYPES


class AnalysisError(ValueError):
    """Only fixed, non-sensitive messages should be raised here."""


def _number(value: Any, duration: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError("Invalid event or frame timestamp")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= duration:
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
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise AnalysisError("Video has no usable timing information")
    result = _mapping(raw)
    verdict, events = result.get("verdict"), result.get("events")
    scoped = result.get("analysis_scope") == "safe_state"
    scenario = result.get("analysis_scope") == "bank_scenarios"
    allowed = ({"review_required", "no_candidate"} if scenario else
               {"review_required", "no_event"} if scoped else {"violation", "no_violation"})
    if not isinstance(verdict, str) or verdict not in allowed or not isinstance(events, list):
        raise AnalysisError("Model result does not match the integration contract")
    if (verdict in {"violation", "review_required"}) != bool(events) or len(events) > 100:
        raise AnalysisError("Model verdict is inconsistent with its events")
    extra = {}
    if scenario:
        extra = _scenario_metadata(result, events, duration)
    if scoped:
        observations = result.get("observations")
        if not isinstance(observations, list) or not 1 <= len(observations) <= 1200:
            raise AnalysisError("Safe analysis requires sampled observations")
        cleaned = []
        for observation in observations:
            if not isinstance(observation, dict) or observation.get("state") not in {"open", "closed", "unknown"}:
                raise AnalysisError("Invalid safe state observation")
            confidence = _confidence(observation.get("confidence"))
            timestamp = _number(observation.get("timecode_sec"), duration)
            if cleaned and timestamp < cleaned[-1]["timecode_sec"]:
                raise AnalysisError("Observations must be in time order")
            cleaned.append({"timecode_sec": timestamp, "state": observation["state"], "confidence": confidence})
        if (verdict == "no_event") != all(item["state"] == "closed" for item in cleaned):
            raise AnalysisError("Safe verdict is inconsistent with sampled observations")
        summary = result.get("summary")
        if isinstance(summary, dict):
            counts = {state: sum(item["state"] == state for item in cleaned) for state in ("open", "closed", "unknown")}
            if summary.get("sample_count") != len(cleaned) or summary.get("state_counts") != counts:
                raise AnalysisError("Summary does not match sampled observations")
            sample_fps = summary.get("sample_fps")
            if isinstance(sample_fps, bool) or not isinstance(sample_fps, (int, float)) or not math.isfinite(sample_fps) or not 0 < sample_fps <= 30:
                raise AnalysisError("Invalid sampling frequency")
            if not isinstance(summary.get("events_truncated"), bool):
                raise AnalysisError("Invalid event truncation flag")
            if summary.get("score_semantics") != "uncalibrated_model_score":
                raise AnalysisError("Invalid model score semantics")
            summary = {"sample_count": len(cleaned), "state_counts": counts, "sample_fps": sample_fps,
                       "events_truncated": summary["events_truncated"], "score_semantics": "uncalibrated_model_score"}
        else:
            summary = _text(summary)
        extra = {"analysis_scope": "safe_state", "model_version": _text(result.get("model_version")),
                 "summary": summary, "observations": cleaned}
    violations = []
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
        if scoped or scenario:
            start = _number(event.get("start_sec"), duration)
            end = _number(event.get("end_sec"), duration)
            if not start <= violation["timecode_sec"] <= end:
                raise AnalysisError("Invalid event interval")
            if any(not start <= frame["timestamp_sec"] <= end for frame in normalized_frames):
                raise AnalysisError("Evidence timestamp is outside its event interval")
            violation.update(start_sec=start, end_sec=end, confidence=_confidence(event.get("confidence")))
        violations.append(violation)
    return {"verdict": verdict, "violations": violations, **extra}


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise AnalysisError("Invalid model confidence")
    return float(value)


def _count(value: Any, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or (maximum is not None and value > maximum):
        raise AnalysisError("Invalid scenario sample or support count")
    return value


def _scenario_metadata(result: dict, events: list, duration: float) -> dict:
    """Validate partial-support candidate scores without asserting a violation."""
    if result.get("score_semantics") != "uncalibrated_model_score":
        raise AnalysisError("Invalid model score semantics")
    policy = result.get("model_support_policy", {"min_positive_media_groups": 1, "min_negative_media_groups": 1})
    if (not isinstance(policy, dict)
            or set(policy) != {"min_positive_media_groups", "min_negative_media_groups"}
            or type(policy.get("min_positive_media_groups")) is not int
            or policy["min_positive_media_groups"] not in (1, 2)
            or type(policy.get("min_negative_media_groups")) is not int
            or policy["min_negative_media_groups"] != 1):
        raise AnalysisError("Invalid scenario support policy")
    classes = result.get("class_results")
    if not isinstance(classes, list) or not 1 <= len(classes) <= 50:
        raise AnalysisError("Scenario analysis requires class results")
    cleaned, seen, candidates = [], set(), set()
    for row in classes:
        if not isinstance(row, dict):
            raise AnalysisError("Invalid scenario class result")
        event_type = _text(row.get("type"))
        if event_type not in SCENARIO_TYPES or event_type in seen:
            raise AnalysisError("Unknown or duplicate scenario type")
        seen.add(event_type)
        threshold = _confidence(row.get("threshold"))
        positive = _count(row.get("positive_groups"))
        negative = _count(row.get("negative_groups"))
        decision = row.get("decision")
        if not isinstance(decision, str) or decision not in {"candidate", "not_flagged", "unsupported"}:
            raise AnalysisError("Invalid scenario decision")
        supported = positive >= policy["min_positive_media_groups"] and negative >= policy["min_negative_media_groups"]
        if (decision != "unsupported") != supported:
            raise AnalysisError("Scenario decision contradicts model support")
        score, peak = row.get("score"), row.get("peak_sec")
        if not supported:
            if score is not None or peak is not None:
                raise AnalysisError("Unsupported scenarios require null scores and times")
        else:
            score, peak = _confidence(score), _number(peak, duration)
            if (decision == "candidate") != (score >= threshold):
                raise AnalysisError("Scenario decision contradicts its threshold")
        if decision == "candidate":
            candidates.add(event_type)
        cleaned.append({"type": event_type, "title": _text(row.get("title")),
                        "score": score, "threshold": threshold, "decision": decision,
                        "positive_groups": positive, "negative_groups": negative, "peak_sec": peak})
    event_type_list = [_text(_mapping(event).get("event_type")) for event in events]
    event_types = set(event_type_list)
    if len(event_type_list) != len(event_types):
        raise AnalysisError("Duplicate scenario events")
    if event_types != candidates or (result["verdict"] == "review_required") != bool(candidates):
        raise AnalysisError("Scenario candidates, events, and verdict are inconsistent")
    by_type = {row['type']: row for row in cleaned}
    for raw_event in events:
        event = _mapping(raw_event)
        row = by_type[event['event_type']]
        if (_confidence(event.get('confidence')) != row['score']
                or _number(event.get('timecode_sec'), duration) != row['peak_sec']):
            raise AnalysisError("Scenario event does not match its model score and peak")
    limitations = result.get("limitations")
    if not isinstance(limitations, list) or not 1 <= len(limitations) <= 50:
        raise AnalysisError("Scenario analysis requires bounded limitations")
    summary = result.get("summary")
    if not isinstance(summary, dict) or not 1 <= _count(summary.get("sample_count"), 1200):
        raise AnalysisError("Scenario analysis requires sampled frames")
    sample_fps = summary.get("sample_fps")
    if isinstance(sample_fps, bool) or not isinstance(sample_fps, (int, float)) or not math.isfinite(sample_fps) or not 0 < sample_fps <= 30:
        raise AnalysisError("Invalid sampling frequency")
    if not isinstance(summary.get("events_truncated"), bool):
        raise AnalysisError("Invalid event truncation flag")
    extra = {"analysis_scope": "bank_scenarios", "model_version": _text(result.get("model_version")),
            "feature_version": _text(result.get("feature_version")),
            "score_semantics": "uncalibrated_model_score", "class_results": cleaned,
            "model_support_policy": dict(policy),
            "limitations": [_text(value) for value in limitations],
            "summary": {"sample_count": summary["sample_count"], "sample_fps": float(sample_fps),
                        "events_truncated": summary["events_truncated"]}}
    if 'coverage' in result:
        extra['coverage'] = _coverage_metadata(result['coverage'], duration)
    return extra


def _coverage_metadata(raw: Any, duration: float) -> dict:
    """Retain only bounded decoder facts; jury projection enforces completeness."""
    if not isinstance(raw, dict) or type(raw.get('full_decode')) is not bool:
        raise AnalysisError("Scenario decoder coverage is missing or invalid")
    times = raw.get('sample_timestamps_sec')
    if not isinstance(times, list) or not 1 <= len(times) <= 1200:
        raise AnalysisError("Scenario decoder sample timestamps are invalid")
    times = [_number(value, duration) for value in times]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise AnalysisError("Scenario decoder sample timestamps are unordered")
    fps = raw.get('sample_fps')
    if (isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not math.isfinite(fps) or not 0 < fps <= 30):
        raise AnalysisError("Invalid sampling frequency")
    result = {key: _number(raw.get(key), duration) for key in (
        'first_sample_sec', 'last_sample_sec', 'last_frame_sec', 'duration_sec', 'max_frame_gap_sec')}
    result.update(full_decode=raw['full_decode'], sample_fps=float(fps),
                  decoded_frame_count=_count(raw.get('decoded_frame_count')),
                  sample_count=_count(raw.get('sample_count'), 1200), sample_timestamps_sec=times)
    return result


def _selected_analyzer(video_path: str, settings: Any) -> tuple[str, str | None]:
    """Read only a private allowlisted selection; paths remain server-owned."""
    sidecar = Path(str(video_path) + ".analysis.json")
    try:
        encoded = sidecar.read_text(encoding="utf-8")
    except FileNotFoundError:
        return getattr(settings, "analyzer_callable", ""), getattr(settings, "model_path", None)
    except (OSError, UnicodeError):
        raise AnalysisError("Cannot read the selected analysis type") from None
    try:
        selection = json.loads(encoded) if len(encoded) <= 256 else None
        if not isinstance(selection, dict) or set(selection) != {"analysis_type"}:
            raise ValueError
        analysis_type = selection["analysis_type"]
        if not isinstance(analysis_type, str) or analysis_type not in ANALYSIS_CALLABLES:
            raise ValueError
    except (ValueError, TypeError):
        raise AnalysisError("Invalid selected analysis type") from None
    if settings.analyzer_mode != "real":
        raise AnalysisError("Analyzer selection requires real mode")
    model_path = getattr(settings, "scenario_model_path", None) if analysis_type == "bank_scenarios" else settings.model_path
    if not model_path:
        raise AnalysisError("Selected local model is not configured")
    return ANALYSIS_CALLABLES[analysis_type], model_path


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
    analyzer_callable, model_path = _selected_analyzer(video_path, settings)
    jury = False
    if getattr(settings, 'jury_analyzer', None) == 'bank_scenarios':
        try:
            mode_path = Path(str(video_path) + '.api-mode.json')
            mode = json.loads(mode_path.read_text(encoding='utf-8'))
            if (not isinstance(mode, dict) or set(mode) != {'analysis_type'}
                    or mode['analysis_type'] not in {'jury', 'bank_scenarios', 'safe_state'}):
                raise ValueError
            jury = mode['analysis_type'] == 'jury'
        except (OSError, UnicodeError, ValueError, TypeError):
            raise AnalysisError('Analysis mode metadata is unavailable') from None
        if jury and (settings.analyzer_mode != 'real'
                     or analyzer_callable != ANALYSIS_CALLABLES['bank_scenarios']
                     or model_path != settings.scenario_model_path):
            raise AnalysisError('Jury route requires the configured local scenario analyzer')
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
        if model_path:
            os.environ["MODEL_PATH"] = str(model_path)
        try:
            module_name, function_name = analyzer_callable.split(":", 1)
            analyzer = getattr(importlib.import_module(module_name), function_name)
        except (ValueError, ImportError, AttributeError):
            raise AnalysisError("Local model is not installed. Configure ANALYZER_CALLABLE and MODEL_PATH") from None
        try:
            raw = analyzer(str(video_path))
        except Exception as exc:
            if jury:
                from .scenario_classifier import ScenarioModelError
                if isinstance(exc, ScenarioModelError):
                    raise AnalysisError('Jury scenario model could not be loaded; check its local artifacts') from None
            raise AnalysisError("Local model failed to analyze the video") from None
        rules = load_rules(settings.rules_path)
    if jury:
        # Presentation timestamps from the model's complete AV decode are the
        # authoritative clock for its samples and evidence (also for VFR MOV).
        coverage = raw.get('coverage') if isinstance(raw, dict) else None
        if not isinstance(coverage, dict):
            raise AnalysisError('Scenario decoder coverage is missing or invalid')
        duration = coverage.get('duration_sec')
    result = normalize_result(raw, duration, rules)
    result["demo"] = settings.analyzer_mode == "demo"
    if jury:
        from .jury_projection import project_scenario_result

        result = project_scenario_result(result, duration)
        try:
            with Path(video_path).open('rb') as source:
                input_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
        except OSError:
            raise AnalysisError('Cannot verify the uploaded video') from None
        result['video_duration_sec'] = duration
        result['provenance'] = {
            'actual_analyzer': analyzer_callable,
            'analysis_type': 'bank_scenarios',
            'execution': 'local',
            'input_sha256': input_sha256,
            'model_artifact_sha256': raw['model_artifact_sha256'],
            'backbone_sha256': raw['backbone_sha256'],
        }
    return result
