"""The standalone checker rejects contract leaks without using private media."""
import base64
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_jury_api as checker


@pytest.fixture
def video_info():
    return {"duration_sec": 3.0, "frame_tolerance_sec": 1 / 30,
            "input_sha256": hashlib.sha256(b"synthetic movie").hexdigest(),
            "decoded_source_frames": 90}


@pytest.fixture
def result(video_info):
    ok, jpeg = cv2.imencode(".jpg", np.full((24, 32, 3), 80, np.uint8))
    assert ok
    classes = [{"type": name, "title": "Synthetic class", "score": 0.2,
                "threshold": 0.65, "decision": "not_flagged", "peak_sec": 1.0,
                "positive_groups": 2, "negative_groups": 2}
               for name in sorted(checker.SCENARIO_TYPES)]
    chosen = classes[0]
    chosen.update(score=0.8, decision="candidate")
    return {
        "job_id": "synthetic-job", "status": "done", "verdict": "violation",
        "demo": False, "analysis_scope": "bank_scenarios", "output_contract": "jury-v1",
        "jury_policy": "scenario-threshold-v1", "model_version": "synthetic-v1",
        "video_duration_sec": 3.0, "score_semantics": "uncalibrated_model_score",
        "model_support_policy": {"min_positive_media_groups": 1, "min_negative_media_groups": 1},
        "class_results": classes,
        "coverage": {"full_decode": True, "decoded_frame_count": 90, "sample_count": 3,
                     "sample_fps": 1.0, "first_sample_sec": 0.0, "last_sample_sec": 2.0,
                     "last_frame_sec": 89 / 30, "duration_sec": 3.0,
                     "max_frame_gap_sec": 1 / 30, "sample_timestamps_sec": [0.0, 1.0, 2.0]},
        "provenance": {"actual_analyzer": checker.ACTUAL_ANALYZER,
                       "input_sha256": video_info["input_sha256"], "execution": "local",
                       "analysis_type": "bank_scenarios", "model_artifact_sha256": "a" * 64,
                       "backbone_sha256": "b" * 64},
        "violations": [{"type": chosen["type"], "description": "Synthetic candidate only.",
                        "timecode_sec": 1.0, "start_sec": 0.0, "end_sec": 2.0,
                        "confidence": 0.8, "evidence_frames": [{"timestamp_sec": 1.0,
                            "image_base64": base64.b64encode(jpeg).decode("ascii")}]}],
    }


def test_valid_positive_and_fully_supported_negative(result, video_info):
    report = checker.validate_result(result, "synthetic-job", video_info)
    assert report["valid_jpeg_frames"] == 1
    assert report["event_count"] == 1
    assert report["quality_validated"] is False
    assert "description" not in json.dumps(report)
    assert "image_base64" not in json.dumps(report)
    result.update(verdict="no_violation", violations=[])
    result["class_results"][0].update(score=0.2, decision="not_flagged")
    report = checker.validate_result(result, "synthetic-job", video_info)
    assert report["valid_jpeg_frames"] == report["event_count"] == 0


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(verdict="review_required"),
    lambda r: r.update(verdict="no_candidate"),
    lambda r: r.update(verdict="no_violation"),
    lambda r: r.update(demo=True),
    lambda r: r.update(job_id="another-job"),
    lambda r: r["provenance"].update(input_sha256="c" * 64),
    lambda r: r["provenance"].update(actual_analyzer="tests.fake_analyzer"),
    lambda r: r["coverage"].update(full_decode=False),
    lambda r: r["coverage"].update(max_frame_gap_sec=2.0),
    lambda r: r["violations"][0].update(timecode_sec=float("nan")),
    lambda r: r["violations"][0].update(timecode_sec=4.0),
    lambda r: r["violations"][0].update(timecode_sec=True),
    lambda r: r["violations"][0].update(evidence_frames=[]),
    lambda r: r["violations"][0].update(evidence_frames=r["violations"][0]["evidence_frames"] * 4),
    lambda r: r["violations"][0]["evidence_frames"][0].update(image_base64="not base64"),
    lambda r: r["violations"][0]["evidence_frames"][0].update(image_base64=base64.b64encode(b"\xff\xd8fake\xff\xd9").decode()),
    lambda r: r["violations"][0]["evidence_frames"][0].update(timestamp_sec=2.5),
    lambda r: r["violations"][0]["evidence_frames"][0].update(timestamp_sec=0.0),
    lambda r: r["class_results"][0].update(score=0.4),
    lambda r: r["class_results"].pop(),
    lambda r: r.update(extra={"nested": [float("inf")]}),
])
def test_invalid_contract_and_evidence_are_rejected(result, video_info, mutation):
    mutation(result)
    with pytest.raises(checker.CheckError):
        checker.validate_result(result, "synthetic-job", video_info)


def test_negative_does_not_hide_unsupported_class(result, video_info):
    result.update(verdict="no_violation", violations=[])
    result["class_results"][0].update(score=None, peak_sec=None, decision="unsupported", positive_groups=0)
    with pytest.raises(checker.CheckError, match="unsupported_negative_decision"):
        checker.validate_result(result, "synthetic-job", video_info)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000/", "http://localhost:8000", "http://[::1]:8000"])
def test_loopback_urls_allowed(url):
    assert checker.endpoint(url) == url.rstrip("/")


@pytest.mark.parametrize("url", ["https://example.com", "http://example.com", "http://user:secret@127.0.0.1",
                                 "http://127.0.0.1/?key=secret", "file:///private", "http://127.0.0.1/path"])
def test_remote_and_credential_urls_rejected_without_explicit_permission(url):
    with pytest.raises(checker.CheckError):
        checker.endpoint(url)


def test_explicit_remote_option_still_requires_https():
    assert checker.endpoint("https://example.com", True) == "https://example.com"
    with pytest.raises(checker.CheckError, match="requires_https"):
        checker.endpoint("http://example.com", True)


class Response:
    def __init__(self, payload, status_code=200):
        self.payload, self.status_code, self.closed = payload, status_code, False

    def iter_content(self, chunk_size):
        yield json.dumps(self.payload).encode()

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses, on_request=lambda: None):
        self.responses, self.calls, self.on_request = responses, [], on_request

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        self.on_request()
        return self.responses.pop(0)


def setup_check(tmp_path, monkeypatch, video_info):
    movie = tmp_path / "synthetic.mov"
    movie.write_bytes(b"synthetic movie")
    monkeypatch.setattr(checker, "inspect_video", lambda path: video_info)
    return movie


def test_http_uses_only_file_and_env_key_no_redirects_or_proxy(tmp_path, monkeypatch, result, video_info):
    movie = setup_check(tmp_path, monkeypatch, video_info)
    session = Session([Response({"job_id": "synthetic-job"}, 202),
                       Response({"job_id": "synthetic-job", "status": "processing"}), Response(result)])
    report = checker.check_api("http://127.0.0.1:8000", movie, "secret-only-in-header", session=session, sleep=lambda _: None)
    assert report["submitted_parameters"] == ["file"]
    assert session.trust_env is False
    assert [item[0] for item in session.calls] == ["POST", "GET", "GET"]
    for _, _, kwargs in session.calls:
        assert kwargs["headers"] == {"X-API-Key": "secret-only-in-header"}
        assert kwargs["allow_redirects"] is False
        assert "data" not in kwargs and "params" not in kwargs
    assert set(session.calls[0][2]["files"]) == {"file"}
    assert "secret-only-in-header" not in json.dumps(report)


@pytest.mark.parametrize("response,code", [
    (Response({"job_id": "synthetic-job", "status": "failed", "error": "private error"}), "analysis_failed"),
    (Response({"job_id": "another-job", "status": "processing"}), "job_id_mismatch"),
    (Response({}, 302), "redirect_rejected"),
    (Response({}, 401), "api_key_rejected"),
])
def test_http_failures_are_not_success_or_raw_response_leaks(tmp_path, monkeypatch, video_info, response, code):
    movie = setup_check(tmp_path, monkeypatch, video_info)
    session = Session([Response({"job_id": "synthetic-job"}, 202), response])
    with pytest.raises(checker.CheckError, match=code) as error:
        checker.check_api("http://127.0.0.1:8000", movie, "private-key", session=session)
    assert "private" not in str(error.value)


def test_deadline_includes_post_upload(tmp_path, monkeypatch, video_info):
    movie = setup_check(tmp_path, monkeypatch, video_info)
    now = [0.0]
    def slow_upload():
        now[0] += 301
    session = Session([Response({"job_id": "synthetic-job"}, 202)], slow_upload)
    with pytest.raises(checker.CheckError, match="deadline_exceeded"):
        checker.check_api("http://127.0.0.1:8000", movie, "private-key", session=session, clock=lambda: now[0])
    assert len(session.calls) == 1


@pytest.mark.parametrize("timeout", [0, -1, 301, float("inf"), float("nan")])
def test_client_timeout_cannot_exceed_case_limit(timeout):
    with pytest.raises(checker.CheckError):
        checker.check_api("http://127.0.0.1:8000", Path("unused.mov"), "private-key", timeout)


def test_missing_env_key_is_safe_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("API_KEY", raising=False)
    private_path = tmp_path / "private-name.mov"
    assert checker.main(["--video", str(private_path)]) == 1
    output = capsys.readouterr().out
    assert json.loads(output)["error"] == "api_key_environment_missing"
    assert "private-name" not in output


def test_cli_writes_only_aggregate_report_and_never_overwrites(tmp_path, monkeypatch, capsys):
    report = {"status": "done", "verdict": "no_violation", "quality_validated": False}
    output = tmp_path / "report.json"
    monkeypatch.setenv("API_KEY", "private-key")
    seen = []
    def fake_check(url, video, key, timeout):
        seen.append((url, video, key, timeout))
        return report
    monkeypatch.setattr(checker, "check_api", fake_check)
    arguments = ["--video", str(tmp_path / "private-video.mov"), "--output", str(output)]
    assert checker.main(arguments) == 0
    assert seen[0][2] == "private-key"
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report
    assert checker.main(arguments) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "local_io_failed"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_actual_local_mov_probe_uses_decode_and_hash(tmp_path):
    movie = tmp_path / "synthetic.mov"
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((24, 32, 3), index, np.uint8))
    writer.release()
    info = checker.inspect_video(movie)
    assert info["decoded_source_frames"] == 20
    assert info["duration_sec"] == pytest.approx(2)
    assert info["input_sha256"] == hashlib.sha256(movie.read_bytes()).hexdigest()
