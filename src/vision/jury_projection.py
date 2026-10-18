"""Project validated local scenario candidates into the jury response contract.

The binary decision describes thresholded model indicators, not a proven
procedural violation. The model's saved thresholds and class support policy are
preserved. Missing support may accompany a positive indicator, but must never
turn an unassessed class into a negative decision.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import math
from typing import Any

import cv2
import numpy as np

from .scenario_contracts import SCENARIO_TYPES
from .scenario_features import SAMPLE_FPS, SUPPORTED_FEATURE_VERSIONS

_EPS = 1e-6
_BINARY_LIMITATION = (
    "Бинарное решение отражает наличие кандидатов выше сохранённых порогов модели "
    "в поддержанных классах на выборке 1 FPS. Оно не подтверждает нарушение "
    "или отсутствие всех нарушений; короткие и невидимые действия могут быть пропущены."
)


def _fail(message: str) -> None:
    # normalize_result imports this projection only for the jury path.
    from .adapter import AnalysisError
    raise AnalysisError(message)


def _finite(value: Any, message: str, *, minimum: float = 0., maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(message)
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        _fail(message)
    return value


def _count(value: Any, message: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        _fail(message)
    return value


def _same(first: float, second: float) -> bool:
    return abs(first - second) <= _EPS


def _coverage(result: dict, duration: float) -> list[float]:
    coverage, summary = result.get("coverage"), result.get("summary")
    message = "Jury analysis requires complete verified video coverage"
    if not isinstance(coverage, dict) or coverage.get("full_decode") is not True or not isinstance(summary, dict):
        _fail(message)
    if summary.get("events_truncated") is not False:
        _fail("Jury analysis cannot publish truncated events")
    decoded = _count(coverage.get("decoded_frame_count"), message)
    count = _count(coverage.get("sample_count"), message)
    if count > decoded or count != _count(summary.get("sample_count"), message):
        _fail(message)
    fps = _finite(coverage.get("sample_fps"), message)
    if fps != SAMPLE_FPS or fps != _finite(summary.get("sample_fps"), message):
        _fail(message)
    measured_duration = _finite(coverage.get("duration_sec"), message)
    if not _same(measured_duration, duration):
        _fail(message)
    first = _finite(coverage.get("first_sample_sec"), message, maximum=duration)
    last = _finite(coverage.get("last_sample_sec"), message, maximum=duration)
    last_frame = _finite(coverage.get("last_frame_sec"), message, maximum=duration)
    frame_gap = _finite(coverage.get("max_frame_gap_sec"), message)
    # A missing whole sampling period cannot justify a complete negative scan.
    period = 1. / fps
    if (not _same(first, 0.) or last > last_frame + _EPS
            or frame_gap > period + _EPS or last_frame - last > period + frame_gap + _EPS
            or duration - last > period + frame_gap + _EPS):
        _fail(message)
    raw_times = coverage.get("sample_timestamps_sec")
    if not isinstance(raw_times, list) or len(raw_times) != count:
        _fail(message)
    times = [_finite(value, message, maximum=duration) for value in raw_times]
    if not _same(times[0], first) or not _same(times[-1], last):
        _fail(message)
    if any(right <= left or right - left > period + frame_gap + _EPS
           for left, right in zip(times, times[1:])):
        _fail(message)
    return times


def _sample_time(value: Any, duration: float, samples: list[float]) -> float:
    timestamp = _finite(value, "Jury event time is outside the video", maximum=duration)
    if not any(_same(timestamp, sample) for sample in samples):
        _fail("Jury event evidence must use analyzed video samples")
    return timestamp


def _jpeg_frames(event: dict, start: float, peak: float, end: float,
                 duration: float, samples: list[float]) -> None:
    frames = event.get("evidence_frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
        _fail("Each jury event requires 1 to 3 JPEG evidence frames")
    timestamps = []
    for frame in frames:
        if not isinstance(frame, dict):
            _fail("Invalid jury evidence frame")
        timestamp = _sample_time(frame.get("timestamp_sec"), duration, samples)
        if not start <= timestamp <= end or any(_same(timestamp, previous) for previous in timestamps):
            _fail("Jury evidence times must be distinct and inside the event interval")
        timestamps.append(timestamp)
        encoded = frame.get("image_base64")
        if not isinstance(encoded, str) or len(encoded) > 4 * ((4 * 1024 * 1024 + 2) // 3):
            _fail("Invalid jury JPEG encoding")
        try:
            image = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            _fail("Invalid jury JPEG encoding")
        if (not 4 <= len(image) <= 4 * 1024 * 1024
                or not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9")):
            _fail("Jury evidence must contain a JPEG image")
        try:
            decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            _fail("Jury evidence JPEG cannot be decoded")
        if decoded is None or min(decoded.shape[:2]) < 1:
            _fail("Jury evidence JPEG cannot be decoded")
    if not any(_same(peak, timestamp) for timestamp in timestamps):
        _fail("Jury evidence must include the scored peak frame")


def project_scenario_result(normalized: dict, duration: float) -> dict:
    """Return the jury projection without mutating the internal model result.

    Run after adapter.normalize_result. Its validated metadata is retained;
    this boundary additionally checks catalogue completeness, decoded coverage,
    candidate/event correspondence, and evidence against analyzed timestamps.
    """
    duration = _finite(duration, "Jury video duration must be finite and positive")
    if duration <= 0:
        _fail("Jury video duration must be finite and positive")
    if not isinstance(normalized, dict) or normalized.get("analysis_scope") != "bank_scenarios":
        _fail("Jury mode requires the local bank scenario analyzer")
    if normalized.get("score_semantics") != "uncalibrated_model_score":
        _fail("Jury mode requires explicit model score semantics")
    if normalized.get("feature_version") not in SUPPORTED_FEATURE_VERSIONS:
        _fail("Jury mode requires a supported scenario feature version")
    if not isinstance(normalized.get("model_version"), str) or not normalized["model_version"].strip():
        _fail("Jury mode requires a model version")
    samples = _coverage(normalized, duration)
    classes = normalized.get("class_results")
    if not isinstance(classes, list) or len(classes) != len(SCENARIO_TYPES):
        _fail("Jury analysis requires the complete scenario catalogue")
    policy = normalized.get("model_support_policy")
    if (not isinstance(policy, dict)
            or type(policy.get("min_positive_media_groups")) is not int
            or policy["min_positive_media_groups"] not in (1, 2)
            or type(policy.get("min_negative_media_groups")) is not int
            or policy["min_negative_media_groups"] != 1):
        _fail("Jury analysis requires the saved model support policy")
    by_type, candidates, unsupported = {}, {}, []
    for row in classes:
        if not isinstance(row, dict):
            _fail("Invalid jury scenario class")
        kind = row.get("type")
        if not isinstance(kind, str) or kind not in SCENARIO_TYPES or kind in by_type:
            _fail("Jury analysis requires unique known scenario classes")
        by_type[kind] = row
        positive = _count(row.get("positive_groups"), "Invalid model class support", minimum=0)
        negative = _count(row.get("negative_groups"), "Invalid model class support", minimum=0)
        supported = positive >= policy["min_positive_media_groups"] and negative >= policy["min_negative_media_groups"]
        decision = row.get("decision")
        threshold = _finite(row.get("threshold"), "Invalid saved model threshold", maximum=1.)
        title = row.get("title")
        if not isinstance(title, str) or not title.strip() or len(title) > 4000:
            _fail("Invalid scenario title")
        if not supported:
            if decision != "unsupported" or row.get("score") is not None or row.get("peak_sec") is not None:
                _fail("Unsupported scenario must retain an explicit unknown result")
            unsupported.append(kind)
            continue
        score = _finite(row.get("score"), "Invalid model score", maximum=1.)
        _sample_time(row.get("peak_sec"), duration, samples)
        expected = "candidate" if score >= threshold else "not_flagged"
        if decision != expected:
            _fail("Scenario decision contradicts its saved threshold")
        if expected == "candidate":
            candidates[kind] = row
    events = normalized.get("violations")
    if not isinstance(events, list) or len(events) != len(candidates):
        _fail("Jury events must correspond exactly to model candidates")
    internal_verdict = normalized.get("verdict")
    if internal_verdict != ("review_required" if candidates else "no_candidate"):
        _fail("Scenario verdict contradicts model candidates")
    if not candidates and unsupported:
        _fail("No binary negative decision is available: some scenario classes are unsupported")
    output_events, seen = [], set()
    for event in events:
        if not isinstance(event, dict):
            _fail("Invalid jury event")
        kind = event.get("type")
        if not isinstance(kind, str) or kind not in candidates or kind in seen:
            _fail("Jury events must correspond exactly to model candidates")
        seen.add(kind)
        row = candidates[kind]
        start = _sample_time(event.get("start_sec"), duration, samples)
        peak = _sample_time(event.get("timecode_sec"), duration, samples)
        end = _sample_time(event.get("end_sec"), duration, samples)
        score = _finite(event.get("confidence"), "Invalid jury event score", maximum=1.)
        if not start <= peak <= end or not _same(peak, row["peak_sec"]) or score != row["score"]:
            _fail("Jury event must preserve its model peak, interval and score")
        _jpeg_frames(event, start, peak, end, duration, samples)
        projected = deepcopy(event)
        projected["description"] = (
            f"Модель выделила окно {start:g}–{end:g} с для класса «{row['title']}». "
            f"Пик сходства с обучающими примерами: {peak:g} с; оценка {score:.6g} "
            f"при сохранённом пороге {row['threshold']:.6g}. "
            "Эта оценка не является вероятностью доказанного нарушения. "
            "Кадры показывают выбранный фрагмент; фактическую причину и нормативное основание проверяет человек."
        )
        output_events.append(projected)
    limitations = normalized.get("limitations")
    if not isinstance(limitations, list) or not limitations or any(not isinstance(item, str) or not item.strip() for item in limitations):
        _fail("Jury analysis requires the model limitations")
    output = deepcopy(normalized)
    output.update(verdict="violation" if output_events else "no_violation",
                  violations=output_events, internal_verdict=internal_verdict,
                  output_contract="jury-v1", jury_policy="scenario-threshold-v1")
    output["limitations"].append(_BINARY_LIMITATION)
    if unsupported:
        output["limitations"].append(
            "Найденные кандидаты относятся только к поддержанным классам. "
            "Не оценены классы: " + ", ".join(sorted(unsupported)) + "."
        )
    output["rules_status"] = (
        "verified_sources_present" if any(item.get("rule_status") == "verified" for item in output_events)
        else "unverified"
    )
    return output
