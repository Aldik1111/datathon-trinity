"""Check the real jury API with one local MOV; print aggregate results only.

The API key is read only from API_KEY. No mode, ROI, callable or model path is
sent. Media stays on loopback unless the operator explicitly allows a remote
HTTPS endpoint. This script verifies integration, not recognition accuracy.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import quote, urlsplit

import av
import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision.scenario_contracts import SCENARIO_TYPES


ACTUAL_ANALYZER = "src.vision.scenario_classifier:analyze_video"
MAX_RESPONSE_BYTES = 180 * 1024 * 1024
TOKEN = re.compile(r"[A-Za-z0-9_.-]{1,160}\Z")


class CheckError(ValueError):
    """A fixed code, never a raw server response or exception message."""


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise CheckError("non_finite_json_number")
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _number(value, *, minimum=0.0, maximum=math.inf):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and minimum <= value <= maximum)
    except OverflowError:
        valid = False
    if not valid:
        raise CheckError("invalid_number_or_timestamp")
    return float(value)


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise CheckError("invalid_required_text")
    return value


def endpoint(value: str, allow_remote: bool = False) -> str:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise CheckError("invalid_api_url") from None
    if (not host or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")
            or parsed.scheme not in {"http", "https"} or port == 0):
        raise CheckError("invalid_api_url")
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if not loopback and not allow_remote:
        raise CheckError("remote_endpoint_requires_allow_remote")
    if not loopback and parsed.scheme != "https":
        raise CheckError("remote_endpoint_requires_https")
    return value.rstrip("/")


def inspect_video(path: Path) -> dict:
    """Decode locally through EOF and measure relative presentation time."""
    if path.suffix.lower() != ".mov" or not path.is_file():
        raise CheckError("local_mov_required")
    first = previous = None
    last_step = last_duration = 0.0
    count = 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise CheckError("video_stream_missing")
            stream = container.streams.video[0]
            rate = float(stream.average_rate or 0)
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise CheckError("video_timestamp_missing")
                pts = float(frame.pts * frame.time_base)
                if not math.isfinite(pts) or (previous is not None and pts < previous - 1e-6):
                    raise CheckError("video_timestamp_invalid")
                if first is None:
                    first = pts
                if previous is not None and pts > previous:
                    last_step = pts - previous
                previous = pts
                last_duration = float((frame.duration or 0) * frame.time_base)
                count += 1
        if not count:
            raise CheckError("video_has_no_frames")
        step = last_duration if last_duration > 0 else last_step
        if step <= 0 and math.isfinite(rate) and rate > 0:
            step = 1.0 / rate
        duration = float(previous - first + step)
        if not math.isfinite(duration) or duration <= 0:
            raise CheckError("video_duration_invalid")
        checksum = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                checksum.update(chunk)
        return {"duration_sec": duration, "frame_tolerance_sec": max(step, 1e-6),
                "input_sha256": checksum.hexdigest(), "decoded_source_frames": count}
    except CheckError:
        raise
    except (OSError, ValueError, av.error.FFmpegError):
        raise CheckError("local_video_decode_failed") from None


def _coverage_and_classes(result, duration):
    coverage = result.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("full_decode") is not True:
        raise CheckError("complete_coverage_missing")
    samples = coverage.get("sample_timestamps_sec")
    count, decoded = coverage.get("sample_count"), coverage.get("decoded_frame_count")
    if (type(count) is not int or type(decoded) is not int or not 1 <= count <= decoded
            or not isinstance(samples, list) or len(samples) != count):
        raise CheckError("invalid_coverage_counts")
    samples = [_number(value, maximum=duration) for value in samples]
    if samples[0] != 0 or any(b <= a for a, b in zip(samples, samples[1:])):
        raise CheckError("invalid_sample_times")
    if (_number(coverage.get("sample_fps")) != 1.0 or coverage.get("first_sample_sec") != samples[0]
            or coverage.get("last_sample_sec") != samples[-1]
            or abs(_number(coverage.get("duration_sec")) - duration) > 1e-6):
        raise CheckError("invalid_coverage_timing")
    last = _number(coverage.get("last_frame_sec"), maximum=duration)
    gap = _number(coverage.get("max_frame_gap_sec"), maximum=1.0 + 1e-6)
    if last < samples[-1] or last - samples[-1] > 1 + gap + 1e-6:
        raise CheckError("incomplete_sample_coverage")
    classes = result.get("class_results")
    policy = result.get("model_support_policy")
    if (not isinstance(classes, list) or not isinstance(policy, dict)
            or type(policy.get("min_positive_media_groups")) is not int
            or policy["min_positive_media_groups"] not in (1, 2)
            or type(policy.get("min_negative_media_groups")) is not int
            or policy.get("min_negative_media_groups") != 1):
        raise CheckError("class_support_metadata_missing")
    rows = {}
    for row in classes:
        if not isinstance(row, dict) or not isinstance(row.get("type"), str):
            raise CheckError("invalid_class_result")
        name = row["type"]
        if name not in SCENARIO_TYPES or name in rows:
            raise CheckError("invalid_class_type")
        positive, negative = row.get("positive_groups"), row.get("negative_groups")
        if type(positive) is not int or type(negative) is not int or min(positive, negative) < 0:
            raise CheckError("invalid_training_support")
        supported = positive >= policy["min_positive_media_groups"] and negative >= 1
        threshold = _number(row.get("threshold"), minimum=0.5, maximum=1)
        if threshold == 0.5:
            raise CheckError("invalid_model_threshold")
        decision = row.get("decision")
        if not supported:
            if decision != "unsupported" or row.get("score") is not None or row.get("peak_sec") is not None:
                raise CheckError("unsupported_class_has_decision")
        else:
            score = _number(row.get("score"), maximum=1)
            _number(row.get("peak_sec"), maximum=duration)
            if decision != ("candidate" if score >= threshold else "not_flagged"):
                raise CheckError("class_decision_threshold_mismatch")
        rows[name] = row
    if set(rows) != set(SCENARIO_TYPES):
        raise CheckError("incomplete_class_catalog")
    if result["verdict"] == "no_violation" and any(row["decision"] != "not_flagged" for row in rows.values()):
        raise CheckError("unsupported_negative_decision")
    if result.get("score_semantics") != "uncalibrated_model_score":
        raise CheckError("invalid_score_semantics")
    return samples, rows


def validate_result(result: dict, job_id: str, video: dict) -> dict:
    """Validate public JSON and return non-sensitive numeric provenance."""
    if not isinstance(result, dict):
        raise CheckError("invalid_json_object")
    _finite_tree(result)
    if result.get("job_id") != job_id or result.get("status") != "done":
        raise CheckError("wrong_job_or_status")
    events = result.get("violations")
    verdict = result.get("verdict")
    if (not isinstance(verdict, str) or verdict not in {"violation", "no_violation"} or not isinstance(events, list)
            or (verdict == "violation") != bool(events) or len(events) > 100):
        raise CheckError("invalid_binary_contract")
    provenance = result.get("provenance")
    if (result.get("demo") is not False or result.get("analysis_scope") != "bank_scenarios"
            or result.get("output_contract") != "jury-v1"
            or result.get("jury_policy") != "scenario-threshold-v1"
            or not isinstance(provenance, dict)
            or provenance.get("actual_analyzer") != ACTUAL_ANALYZER
            or provenance.get("execution") != "local"
            or provenance.get("analysis_type") != "bank_scenarios"):
        raise CheckError("real_scenario_provenance_missing")
    if provenance.get("input_sha256") != video["input_sha256"]:
        raise CheckError("input_checksum_mismatch")
    for name in ("model_artifact_sha256", "backbone_sha256"):
        if not isinstance(provenance.get(name), str) or not re.fullmatch(r"[0-9a-f]{64}", provenance[name]):
            raise CheckError("model_checksum_missing")
    model_version = result.get("model_version")
    if not isinstance(model_version, str) or not TOKEN.fullmatch(model_version):
        raise CheckError("invalid_model_version")
    duration = _number(result.get("video_duration_sec"), minimum=1e-9)
    if abs(duration - video["duration_sec"]) > video["frame_tolerance_sec"] + 1e-6:
        raise CheckError("video_duration_mismatch")
    # The server's duration is authoritative for its timestamps, but must agree
    # with the independently decoded source to within one frame.
    samples, classes = _coverage_and_classes(result, duration)
    valid_frames = 0
    found = set()
    for event in events:
        if not isinstance(event, dict):
            raise CheckError("invalid_event")
        event_type = _text(event.get("type"))
        if event_type in found or event_type not in classes or classes[event_type]["decision"] != "candidate":
            raise CheckError("event_class_mismatch")
        found.add(event_type)
        _text(event.get("description"))
        peak = _number(event.get("timecode_sec"), maximum=duration)
        start = _number(event.get("start_sec"), maximum=duration)
        end = _number(event.get("end_sec"), maximum=duration)
        if not start <= peak <= end:
            raise CheckError("invalid_event_interval")
        score = _number(event.get("confidence"), maximum=1)
        if (abs(score - classes[event_type]["score"]) > 1e-6
                or abs(peak - classes[event_type]["peak_sec"]) > 1e-6):
            raise CheckError("event_class_score_or_peak_mismatch")
        if any(not any(abs(point - sample) <= 1e-6 for sample in samples) for point in (start, peak, end)):
            raise CheckError("event_not_in_sampled_observations")
        frames = event.get("evidence_frames")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 3:
            raise CheckError("invalid_evidence_count")
        timestamps = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise CheckError("invalid_evidence_frame")
            timestamp = _number(frame.get("timestamp_sec"), maximum=duration)
            if not start <= timestamp <= end:
                raise CheckError("evidence_outside_event")
            if not any(abs(timestamp - sample) <= 1e-6 for sample in samples):
                raise CheckError("evidence_not_in_sampled_observations")
            timestamps.append(timestamp)
            encoded = frame.get("image_base64")
            if not isinstance(encoded, str) or len(encoded) > 6 * 1024 * 1024:
                raise CheckError("invalid_jpeg_base64")
            try:
                jpeg = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                raise CheckError("invalid_jpeg_base64") from None
            if (not 4 <= len(jpeg) <= 4 * 1024 * 1024
                    or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
                raise CheckError("invalid_jpeg_markers")
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.ndim != 3 or min(decoded.shape[:2]) <= 0:
                raise CheckError("jpeg_decode_failed")
            valid_frames += 1
        if len(set(timestamps)) != len(timestamps) or not any(abs(t - peak) <= 1e-6 for t in timestamps):
            raise CheckError("evidence_peak_missing_or_duplicated")
    if found != {name for name, row in classes.items() if row["decision"] == "candidate"}:
        raise CheckError("candidate_event_missing")
    return {"status": "done", "verdict": verdict, "model_version": model_version,
            "actual_analyzer": ACTUAL_ANALYZER, "output_contract": "jury-v1",
            "jury_policy": "scenario-threshold-v1", "event_count": len(events),
            "valid_jpeg_frames": valid_frames, "video_duration_sec": duration,
            "input_sha256": video["input_sha256"],
            "model_artifact_sha256": provenance["model_artifact_sha256"],
            "backbone_sha256": provenance["backbone_sha256"],
            "quality_validated": False}


def check_api(url, video_path, api_key, timeout=300.0, *, session=None,
              clock=time.monotonic, sleep=time.sleep) -> dict:
    if not isinstance(api_key, str) or not api_key.strip():
        raise CheckError("api_key_environment_missing")
    _number(timeout, minimum=0.001, maximum=300)
    video_path = Path(video_path)
    video = inspect_video(video_path)
    own_session = session is None
    session = session or requests.Session()
    # Do not forward a local secret/video through proxy environment variables.
    session.trust_env = False
    started = clock()
    deadline = started + timeout
    headers = {"X-API-Key": api_key}

    def remaining():
        value = deadline - clock()
        if value <= 0:
            raise CheckError("five_minute_deadline_exceeded")
        return value

    def request(method, route, expected, **kwargs):
        try:
            response = session.request(method, url + route, headers=headers,
                                       timeout=remaining(), allow_redirects=False,
                                       stream=True, **kwargs)
            try:
                remaining()
                if response.status_code != expected:
                    if response.status_code in (401, 403):
                        raise CheckError("api_key_rejected")
                    if 300 <= response.status_code < 400:
                        raise CheckError("redirect_rejected")
                    raise CheckError("unexpected_http_status")
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    remaining()
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise CheckError("response_too_large")
                    chunks.append(chunk)
                result = json.loads(b"".join(chunks))
                remaining()
                if not isinstance(result, dict):
                    raise CheckError("invalid_json_object")
                _finite_tree(result)
                return result
            finally:
                response.close()
        except requests.RequestException:
            raise CheckError("http_request_failed") from None
        except (ValueError, UnicodeError) as exc:
            if isinstance(exc, CheckError):
                raise
            raise CheckError("invalid_response_json") from None

    try:
        with video_path.open("rb") as source:
            pending = request("POST", "/api/v1/analyze", 202,
                              files={"file": ("input.mov", source, "video/quicktime")})
        job_id = pending.get("job_id")
        if not isinstance(job_id, str) or not TOKEN.fullmatch(job_id):
            raise CheckError("invalid_job_id")
        if "status" in pending and pending["status"] != "processing":
            raise CheckError("invalid_post_status")
        polls = 0
        while True:
            result = request("GET", "/api/v1/jobs/" + quote(job_id, safe=""), 200)
            polls += 1
            if result.get("job_id") != job_id:
                raise CheckError("job_id_mismatch")
            status = result.get("status")
            if status == "failed":
                raise CheckError("analysis_failed")
            if status == "done":
                report = validate_result(result, job_id, video)
                remaining()
                report.update(elapsed_sec=round(clock() - started, 3), polls=polls,
                              submitted_parameters=["file"], timeout_sec=float(timeout))
                return report
            if status != "processing":
                raise CheckError("invalid_job_status")
            sleep(min(0.25, remaining()))
    finally:
        if own_session:
            session.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional local aggregate JSON report; never raw responses")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--allow-remote", action="store_true", help="Explicitly allow uploading to a remote HTTPS endpoint")
    args = parser.parse_args(argv)
    try:
        url = endpoint(args.url, args.allow_remote)
        report = check_api(url, args.video, os.environ.get("API_KEY"), args.timeout)
        if args.output:
            # Refuse accidental overwrite, including a source video or .env.
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 0
    except CheckError as exc:
        print(json.dumps({"status": "check_failed", "error": str(exc), "quality_validated": False}))
        return 1
    except (OSError, cv2.error):
        print(json.dumps({"status": "check_failed", "error": "local_io_failed", "quality_validated": False}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
