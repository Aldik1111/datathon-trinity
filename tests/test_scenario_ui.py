"""Experimental scenario UI: original-video transport, provenance, honest evaluation."""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.ui.client import APIClientError, APISettings, submit_video


APP = Path(__file__).resolve().parents[1] / "src/ui/app.py"
SCENARIO_ENV = {
    "ANALYZER_CALLABLE": "src.vision.safe_classifier:analyze_video",
    "ANALYZER_MODE": "real", "API_KEY": "test-key",
    "SCENARIO_MODEL_PATH": "test-scenario-model",
    "SCENARIO_DATASET_MANIFEST": "", "SCENARIO_COMPARISON_REPORT": "",
}
TYPES = ["cash_bag_unattended", "cash_count_missing", "client_left_before_count_complete",
         "door_left_open", "entry_through_open_door", "identity_check_missing", "mixed_client_cash",
         "mixed_denominations", "money_not_secured", "money_not_secured_before_client_left",
         "money_returned", "parallel_service", "personal_items_before_shift", "protected_items_unsecured",
         "report_record_missing", "safe_key_left", "safe_left_open", "unattended_screen"]


def rendered(at):
    return "\n".join(str(element.value) for element in at if hasattr(element, "value"))


def test_quality_notice_uses_independent_metrics_before_manual_upload(tmp_path):
    report = tmp_path / "quality.json"
    report.write_text(json.dumps({"rows": [], "out_of_fold": {
        "micro": {"tp": 4, "scored_positive": 18, "unsupported": 11},
        "overall_binary": {"fp": 19, "scored_negative": 20}}}), encoding="utf-8")
    with patch.dict(os.environ, {**SCENARIO_ENV, "SCENARIO_COMPARISON_REPORT": str(report)}):
        at = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not at.exception
    notices = "\n".join(str(item.value) for item in at.error)
    assert "4 из 18" in notices
    assert "19 из 20" in notices
    assert "11" in notices


def manifest_42(tmp_path):
    rows = []
    for number in range(1, 43):
        video = tmp_path / f"X-{number}.mov"
        video.write_bytes(f"original-video-{number}".encode())
        rows.append({"row_id": f"X-{number}", "filename": video.name,
                     "original_media": {"path": str(video)}, "annotation_status": "review",
                     "table_type_ids": ["money_returned"], "user_description": "PRIVATE TABLE LABEL"})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return manifest


def result(candidate=False):
    image = BytesIO()
    Image.new("RGB", (8, 16), "blue").save(image, "JPEG")
    classes = [{"type": kind, "title": kind, "score": 0.15, "threshold": 0.65,
                "decision": "not_flagged", "positive_groups": 1, "negative_groups": 19,
                "peak_sec": 3.5} for kind in TYPES]
    classes[-1].update(score=None, peak_sec=None, decision="unsupported", positive_groups=0)
    if candidate:
        classes[0].update(score=0.9, decision="candidate")
    return {"job_id": "scenario-job", "status": "done", "demo": False,
            "analysis_scope": "bank_scenarios", "verdict": "review_required" if candidate else "no_candidate",
            "model_version": "scenario-test-1", "feature_version": "features-test-1",
            "score_semantics": "uncalibrated_model_score", "class_results": classes,
            "summary": {"sample_count": 12, "sample_fps": 1, "events_truncated": False},
            "limitations": ["Weak labels cannot establish the cause of an action."],
            "violations": [{"type": TYPES[0], "timecode_sec": 3.5, "start_sec": 2.0, "end_sec": 5.0,
                            "confidence": 0.9, "description": "Candidate from weak labels",
                            "evidence_frames": [{"timestamp_sec": sec, "image_base64": base64.b64encode(image.getvalue()).decode()}
                                                for sec in (2.0, 3.5, 5.0)]}] if candidate else []}


def test_scenario_client_posts_only_mode_and_original_movie_and_keeps_legacy_request():
    with patch("src.ui.client.requests.post") as post:
        post.return_value = Mock(ok=True, json=lambda: {"job_id": "new", "status": "processing"})
        settings = APISettings("http://127.0.0.1:8000", "test-key")
        submit_video(settings, "original.mov", b"original", analysis_type="bank_scenarios")
        assert post.call_args.kwargs["data"] == {"analysis_type": "bank_scenarios"}
        assert post.call_args.kwargs["files"]["file"] == ("original.mov", b"original", "video/quicktime")
        submit_video(settings, "original.mov", b"original")
        assert "data" not in post.call_args.kwargs


def test_scenario_client_rejects_unknown_mode_before_any_network_request():
    with pytest.raises(APIClientError, match="режим"):
        submit_video(APISettings("http://127.0.0.1:1", "test-key"), "original.mov", b"original", analysis_type="other")


def test_all_42_originals_can_be_selected_and_submitted_without_roi_or_table_labels(tmp_path):
    manifest = manifest_42(tmp_path)
    with patch.dict(os.environ, {**SCENARIO_ENV, "SCENARIO_DATASET_MANIFEST": str(manifest)}), \
            patch("src.ui.client.requests.post") as post, \
            patch("src.ui.client.get_job", return_value={"status": "processing"}):
        post.return_value = Mock(ok=True, json=lambda: {"job_id": "new", "status": "processing"})
        at = AppTest.from_file(str(APP), default_timeout=10).run()
        assert at.radio(key="analysis_mode").value == "bank_scenarios"
        assert len(at.selectbox(key="scenario_example").options) == 42
        assert not at.number_input
        at.selectbox(key="scenario_example").set_value(41).run()
        next(button for button in at.button if button.label == "Анализировать").click().run()
    assert not at.exception
    assert post.call_args.kwargs["files"]["file"] == ("X-42.mov", b"original-video-42", "video/quicktime")
    assert post.call_args.kwargs["data"] == {"analysis_type": "bank_scenarios"}
    assert "PRIVATE TABLE LABEL" not in rendered(at)


@pytest.mark.parametrize("candidate", [False, True])
def test_scenario_result_is_scoped_displays_all_classes_and_never_claims_no_violations(candidate):
    with patch.dict(os.environ, SCENARIO_ENV), patch("src.ui.client.get_job", return_value=result(candidate)):
        at = AppTest.from_file(str(APP), default_timeout=10)
        at.query_params["job_id"] = "scenario-job"
        at.run()
    assert not at.exception
    text = rendered(at)
    assert "Нарушение не обнаружено" not in text
    assert "scenario-test-1" in text and "features-test-1" in text
    assert "не подтверждает" in text
    assert "1 пример" in at.dataframe[0].value.iloc[0]["Поддержка"]
    assert len(at.dataframe[0].value) == 18
    assert "Нет поддержки" in at.dataframe[0].value.to_string()
    assert len(at.image) == (3 if candidate else 0)
    assert len(at.get("download_button")) == 1
    if candidate:
        assert "00:02–00:05" in text
    else:
        assert "порог" in text.lower()


def test_upload_and_mode_switch_keep_result_attributed_to_original_mode_and_filename():
    job = result()
    with patch.dict(os.environ, SCENARIO_ENV), \
            patch("src.ui.client.requests.post") as post:
        post.return_value = Mock(ok=True, json=lambda: job)
        at = AppTest.from_file(str(APP), default_timeout=10).run()
        at.file_uploader[0].set_value(("first.mov", b"first-original", "video/quicktime")).run()
        next(button for button in at.button if button.label == "Анализировать").click().run()
        assert "Результат для видео: first.mov" in rendered(at)
        assert "результат предыдущего запуска" not in rendered(at)
        at.file_uploader[0].set_value(("renamed.mov", b"first-original", "video/quicktime")).run()
        assert "результат предыдущего запуска" in rendered(at)
        at.radio(key="analysis_mode").set_value("safe_state").run()
    assert not at.exception
    assert "Результат для видео: first.mov" in rendered(at)
    assert "Режим результата: Сценарии из таблицы" in rendered(at)
    assert "результат предыдущего запуска" in rendered(at)


def comparison_42(tmp_path):
    rows = []
    for i in range(42):
        rows.append({"row_id": f"X-{i+1}", "filename": f"X-{i+1}.mov", "expected_types": ["money_returned"],
                     "annotation_status": "review" if i < 3 else "reviewed", "duplicate_conflict": i < 2,
                     "out_of_fold": {"evaluation": "leave_one_media_group_out", "candidate_types": [],
                                     "unsupported_types": ["money_returned"], "class_results": []},
                     "in_sample": {"evaluation": "resubstitution", "candidate_types": ["money_returned"],
                                   "unsupported_types": [], "class_results": []}})
    report = {"schema_version": 1, "model_version": "model-1", "feature_version": "feature-1", "rows": rows,
              "summary": {"rows": 42, "groups": 40, "types": 18, "duplicate_conflict_rows": 2},
              "out_of_fold": {"micro": {"precision": 0.0, "recall": 0.0}, "per_type": {}},
              "in_sample": {"micro": {"precision": 1.0, "recall": 1.0}, "per_type": {}}}
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_comparison_uses_readable_titles_source_status_and_saved_full_run_link(tmp_path):
    path = comparison_42(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["classes"] = [{"type": "money_returned", "title": "Возврат денег"}]
    report["rows"][0].update(final_job_id="saved-job-1")
    report["rows"][1].update(table_violation=False, expected_types=[], annotation_status="done")
    path.write_text(json.dumps(report), encoding="utf-8")
    with patch.dict(os.environ, {**SCENARIO_ENV, "SCENARIO_COMPARISON_REPORT": str(path)}):
        at = AppTest.from_file(str(APP), default_timeout=10).run()
        table = at.dataframe[-1].value
        assert table.iloc[0]["Метки таблицы"] == "Возврат денег"
        assert table.iloc[1]["Метки таблицы"] == "Без нарушения по таблице"
        assert table.iloc[0]["Проверка источника"] == "На проверке"
        assert table.iloc[1]["Проверка источника"] == "Размечено"
        at.selectbox(key="scenario_comparison_row").set_value("X-1").run()
    assert not at.exception
    assert at.get("link_button")[0].proto.url == "http://127.0.0.1:8501/?job_id=saved-job-1"


def test_offline_comparison_separates_heldout_and_in_sample_and_keeps_all_rows(tmp_path):
    report = comparison_42(tmp_path)
    with patch.dict(os.environ, {**SCENARIO_ENV, "SCENARIO_COMPARISON_REPORT": str(report)}):
        at = AppTest.from_file(str(APP), default_timeout=10).run()
        assert not at.exception
        assert at.radio(key="scenario_comparison_mode").value == "out_of_fold"
        table = at.dataframe[-1].value
        assert len(table) == 42
        assert table.iloc[0]["Кандидаты модели"] == "—"
        assert "42" in rendered(at) and "Конфликты дублей: 2" in rendered(at)
        assert "На проверке источника: 3" in rendered(at)
        at.radio(key="scenario_comparison_mode").set_value("in_sample").run()
        assert "не независимая оценка" in rendered(at)
        assert at.dataframe[-1].value.iloc[0]["Кандидаты модели"] == "money_returned"
        at.selectbox(key="scenario_comparison_row").set_value("X-42").run()
    assert not at.exception
    assert "X-42.mov" in rendered(at)
    assert len(at.get("download_button")) == 2
