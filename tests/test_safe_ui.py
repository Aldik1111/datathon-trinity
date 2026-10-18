"""Safe-state UI contract: scope, calibrated regions, and original video transport."""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.ui.client import APISettings, submit_video


APP = Path(__file__).resolve().parents[1] / "src/ui/app.py"
SAFE_ENV = {"ANALYZER_CALLABLE": "src.vision.safe_classifier:analyze_video", "API_KEY": "test-key"}


def rendered(at):
    return "\n".join(str(element.value) for element in at if hasattr(element, "value"))


def test_optional_roi_is_transported_as_multipart_json_without_changing_legacy_request():
    settings = APISettings("http://127.0.0.1:8000", "test-key")
    with patch("src.ui.client.requests.post") as post:
        post.return_value = Mock(ok=True, json=lambda: {"job_id": "one", "status": "processing"})
        submit_video(settings, "original.mov", b"original-mov", roi=[0.2, 0.1, 0.3, 0.4])
        assert json.loads(post.call_args.kwargs["data"]["roi"]) == [0.2, 0.1, 0.3, 0.4]
        assert post.call_args.kwargs["files"]["file"][1] == b"original-mov"
        submit_video(settings, "original.mov", b"original-mov")
        assert "data" not in post.call_args.kwargs


@pytest.mark.parametrize(
    ("verdict", "state", "message"),
    [
        ("review_required", "open", "Сейф открыт — проверьте контекст"),
        ("review_required", "unknown", "неопредел"),
        ("no_event", "closed", "Открытый сейф не обнаружен в проверенных кадрах"),
    ],
)
def test_scoped_results_loaded_by_query_never_claim_absence_of_all_violations(verdict, state, message):
    buffer = BytesIO()
    Image.new("RGB", (6, 4), "blue").save(buffer, "JPEG")
    job = {
        "job_id": "known-job", "status": "done", "demo": False,
        "verdict": verdict, "analysis_scope": "safe_state", "model_version": "safe-test-1",
        "summary": "Проверено 3 кадра", "observations": [{"timecode_sec": 1.25, "state": state, "confidence": 0.9}],
        "violations": [] if state == "closed" else [{
            "type": "safe_open" if state == "open" else "safe_unknown", "timecode_sec": 1.25,
            "start_sec": 1.25, "end_sec": 2.5, "description": "Требуется ручная проверка",
            "evidence_frames": [{"timestamp_sec": 1.25, "image_base64": base64.b64encode(buffer.getvalue()).decode()}],
        }],
    }
    with patch.dict(os.environ, SAFE_ENV), patch("src.ui.client.get_job", return_value=job):
        at = AppTest.from_file(str(APP), default_timeout=10)
        at.query_params["job_id"] = "known-job"
        at.run()
    assert not at.exception
    text = rendered(at)
    assert message.lower() in text.lower()
    assert "safe-test-1" in text
    assert "Нарушение не обнаружено" not in text
    assert "Нормативное основание не подтверждено" not in text
    assert len(at.dataframe) >= 1
    assert len(at.get("download_button")) == 1
    assert len(at.image) == (0 if state == "closed" else 1)


def test_known_example_submits_original_mov_with_edited_roi_and_test_split(tmp_path):
    import cv2
    import numpy as np

    video = tmp_path / "heldout.mov"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 2, (64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    original = video.read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"videos": [{
        "path": str(video), "filename": video.name, "roi": [0.2, 0.1, 0.4, 0.5],
        "split": "test", "video_id": "heldout", "label": "open",
    }]}), encoding="utf-8")
    with patch.dict(os.environ, {**SAFE_ENV, "SAFE_DATASET_MANIFEST": str(manifest)}), \
            patch("src.ui.client.requests.post") as post, \
            patch("src.ui.client.get_job", return_value={"status": "processing"}):
        post.return_value = Mock(ok=True, json=lambda: {"job_id": "example-job", "status": "processing"})
        at = AppTest.from_file(str(APP), default_timeout=10).run()
        at.radio[0].set_value("Локальные примеры").run()
        assert "Отложенный тест" in rendered(at)
        assert len(at.image) >= 1
        at.number_input(key="roi_x").set_value(0.3).run()
        at.checkbox(key="roi_confirmed").check().run()
        at.number_input(key="roi_y").set_value(0.2).run()
        assert not at.checkbox(key="roi_confirmed").value
        at.number_input(key="roi_y").set_value(0.1).run()
        at.checkbox(key="roi_confirmed").check().run()
        next(button for button in at.button if button.label == "Анализировать").click().run()
    assert not at.exception
    assert post.call_args.kwargs["files"]["file"][1] == original
    assert json.loads(post.call_args.kwargs["data"]["roi"]) == [0.3, 0.1, 0.4, 0.5]


def test_safe_upload_cannot_submit_without_preview_and_explicit_roi_confirmation():
    with patch.dict(os.environ, SAFE_ENV), patch("src.ui.client.requests.post") as post:
        at = AppTest.from_file(str(APP), default_timeout=10).run()
        at.file_uploader[0].set_value(("bad.mov", b"not-video", "video/quicktime")).run()
        next(button for button in at.button if button.label == "Анализировать").click().run()
    assert not at.exception
    assert post.call_count == 0
    assert "кадр" in rendered(at).lower()


def test_report_shows_heldout_sample_size_coverage_and_camera_limitations(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "splits": {"test": {"samples": 14, "videos": 2, "groups": 2,
            "class_counts": {"open": 6, "closed": 8}, "coverage": 0.5, "abstained": 7,
            "per_class": {"open": {"precision": 0.75, "recall": 0.5, "f1": 0.6, "support": 6}},
            "confusion_matrix": [[7, 1], [3, 3]], "accuracy": 0.7142857}},
        "limitations": ["Only the two source cameras were tested."],
    }), encoding="utf-8")
    with patch.dict(os.environ, {**SAFE_ENV, "SAFE_MODEL_REPORT": str(report)}):
        at = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not at.exception
    text = rendered(at)
    assert "14 кадров" in text
    assert "2 видео" in text
    assert "50.0%" in text
    assert "Only the two source cameras were tested." in text
    assert "не точность обнаружения событий" in text
    assert at.dataframe[0].value.iloc[0]["precision"] == 0.75


def test_changing_selected_video_identifies_result_as_previous_run(tmp_path):
    import cv2
    import numpy as np

    first = tmp_path / "first.mov"
    writer = cv2.VideoWriter(str(first), cv2.VideoWriter_fourcc(*"mp4v"), 2, (64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    second = tmp_path / "second.mov"
    second.write_bytes(first.read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"videos": [{
        "path": str(path), "filename": path.name, "roi": [0.2, 0.1, 0.4, 0.5],
        "split": "test", "video_id": path.stem,
    } for path in (first, second)]}), encoding="utf-8")
    result = {"job_id": "previous", "status": "done", "analysis_scope": "safe_state", "verdict": "no_event",
              "observations": [{"timecode_sec": 0, "state": "closed", "confidence": 0.9}], "violations": []}
    with patch.dict(os.environ, {**SAFE_ENV, "SAFE_DATASET_MANIFEST": str(manifest)}), \
            patch("src.ui.client.get_job", return_value=result), \
            patch("src.ui.client.submit_video", return_value={**result, "job_id": "new-job"}):
        at = AppTest.from_file(str(APP), default_timeout=10)
        at.query_params["job_id"] = "previous"
        at.run()
        at.radio[0].set_value("Локальные примеры").run()
        assert "результат предыдущего запуска" in rendered(at)
        at.checkbox(key="roi_confirmed").check().run()
        next(button for button in at.button if button.label == "Анализировать").click().run()
        assert "Результат для видео: first.mov" in rendered(at)
        assert "результат предыдущего запуска" not in rendered(at)
        at.selectbox[0].set_value(1).run()
    assert not at.exception
    assert "результат предыдущего запуска" in rendered(at)
    assert "Результат для видео: first.mov" in rendered(at)
