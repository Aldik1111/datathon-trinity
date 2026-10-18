"""Streamlit interface for the local video analysis API.

Run from the repository root with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from PIL import Image, ImageDraw
import streamlit as st

from src.ui.client import (
    APIClientError,
    decode_evidence_image,
    format_timecode,
    get_job,
    load_settings,
    submit_video,
)


st.set_page_config(page_title="Проверка видео", page_icon="🎥", layout="centered")


def _show_demo_warning(demo: bool) -> None:
    if demo:
        st.warning(
            "Демонстрационный режим: результат сформирован тестовым ответом и не подтверждает реальное нарушение."
        )


def _render_violation(event: Any, index: int) -> None:
    if not isinstance(event, dict):
        st.warning(f"Событие {index}: данные события имеют неверный формат.")
        return

    title = event.get("type") or f"Событие {index}"
    st.markdown(f"**{title} · {format_timecode(event.get('timecode_sec'))}**")
    description = event.get("description")
    if isinstance(description, str) and description.strip():
        st.write(description)

    rule = event.get("rule")
    if isinstance(rule, dict):
        rule_title = rule.get("title")
        section = rule.get("section")
        url = rule.get("url")
        if rule_title or section:
            text = " · ".join(str(part) for part in (rule_title, section) if part)
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                st.markdown(f"Источник: [{text}]({url})")
            elif text:
                st.caption(f"Источник: {text}")
        st.caption("Связь события с указанной нормой и её применимость требуют ручной проверки.")
    else:
        st.warning("Нормативное основание не подтверждено.")

    frames = event.get("evidence_frames", [])
    if not isinstance(frames, list):
        st.caption("Кадры доказательства имеют неверный формат.")
        return
    valid_frames = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_bytes = decode_evidence_image(frame.get("image_base64"))
        if image_bytes is not None:
            valid_frames.append((frame.get("timestamp_sec"), image_bytes))

    if valid_frames:
        columns = st.columns(min(3, len(valid_frames)))
        for frame_index, (timestamp, image_bytes) in enumerate(valid_frames):
            with columns[frame_index % len(columns)]:
                st.image(image_bytes, caption=f"Кадр · {format_timecode(timestamp)}", width="stretch")
    elif frames:
        st.caption("Кадры доказательства не удалось прочитать.")


def _render_result(job: dict[str, Any]) -> None:
    if job.get("analysis_scope") == "safe_state":
        _render_safe_result(job)
        return
    verdict = job.get("verdict")
    violations = job.get("violations", [])

    if verdict == "violation":
        st.error("Обнаружены признаки возможных нарушений.")
        if not isinstance(violations, list) or not violations:
            st.info("Сервис не приложил описание событий.")
        else:
            for index, event in enumerate(violations, start=1):
                with st.container(border=True):
                    _render_violation(event, index)
    elif verdict == "no_violation":
        st.success("Нарушение не обнаружено.")
    else:
        st.error("API вернул неизвестный итоговый статус.")

    st.info(
        "Результат автоматического анализа является предварительным. Все события и кадры должен проверить человек."
    )


def _render_safe_result(job: dict[str, Any]) -> None:
    observations = job.get("observations", [])
    observations = [row for row in observations if isinstance(row, dict)] if isinstance(observations, list) else []
    events = job.get("violations", [])
    events = [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []
    has_open = any(row.get("state") == "open" for row in observations) or any(
        event.get("type") == "safe_open" for event in events
    )
    has_unknown = any(row.get("state") == "unknown" for row in observations) or any(
        event.get("type") in ("safe_unknown", "safe_uncertain") for event in events
    )
    if job.get("verdict") == "review_required":
        if has_open:
            st.warning("Сейф открыт — проверьте контекст")
        else:
            st.warning("Состояние сейфа неопределённо — требуется ручная проверка")
    elif job.get("verdict") == "no_event" and not has_unknown and not has_open:
        st.success("Открытый сейф не обнаружен в проверенных кадрах")
    else:
        st.warning("Состояние сейфа требует ручной проверки")
    if has_unknown:
        st.warning("Есть неопределённые кадры: проверьте область сейфа и контекст вручную.")
    st.caption(f"Модель: {job.get('model_version', 'версия не указана')}")
    summary = job.get("summary")
    if isinstance(summary, dict):
        counts = summary.get("state_counts", {})
        st.write(f"Проверено кадров: {summary.get('sample_count', len(observations))}")
        if isinstance(counts, dict):
            st.write(f"Открыт: {counts.get('open', 0)} · Закрыт: {counts.get('closed', 0)} · Неопределённо: {counts.get('unknown', 0)}")
        if summary.get("events_truncated"):
            st.warning("Достигнут лимит событий; показаны не все фрагменты.")
    elif isinstance(summary, str):
        st.write(summary)
    if observations:
        st.markdown("**Наблюдения по времени**")
        state_labels = {"open": "Открыт", "closed": "Закрыт", "unknown": "Неопределённо"}
        rows = [{"Время, с": row.get("timecode_sec"), "Состояние": state_labels.get(row.get("state"), "Неопределённо"),
                 "Оценка уверенности": row.get("confidence")} for row in observations]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Оценка уверенности — оценка модели, не калиброванная вероятность правильности. Между проверенными кадрами события могут быть пропущены.")
    for event in events:
        with st.container(border=True):
            start = format_timecode(event.get("start_sec", event.get("timecode_sec")))
            end = format_timecode(event.get("end_sec", event.get("timecode_sec")))
            event_type = event.get("type")
            event_titles = {"safe_open": "Сейф открыт", "safe_uncertain": "Недостаточная уверенность", "safe_unknown": "Недостаточная уверенность"}
            st.markdown(f"**{event_titles.get(event_type, 'Фрагмент для проверки')} · {start}–{end}**")
            if event_type == "safe_open":
                st.write("В выбранной области обнаружены признаки открытого сейфа. Проверьте кадры и контекст вручную.")
            elif event_type in ("safe_uncertain", "safe_unknown"):
                st.write("Модель не смогла уверенно определить состояние сейфа. Проверьте выбранную область и кадры вручную.")
            elif isinstance(event.get("description"), str):
                st.write(event["description"])
            for frame in event.get("evidence_frames", []):
                if isinstance(frame, dict):
                    image = decode_evidence_image(frame.get("image_base64"))
                    if image:
                        st.image(image, caption=f"Кадр · {format_timecode(frame.get('timestamp_sec'))}", width="stretch")
    st.info("Область анализа: только открытое/закрытое состояние выбранного сейфа. Это не проверка личности, передач, замков или соблюдения регламентов. Все отмеченные и неопределённые фрагменты проверяет человек.")
    st.download_button("Скачать результат JSON", json.dumps(job, ensure_ascii=False, indent=2),
                       file_name=f"safe-result-{job.get('job_id', 'analysis')}.json", mime="application/json")


def _valid_roi(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        return False
    x, y, w, h = value
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 1 and y + h <= 1


def _local_examples() -> list[dict[str, Any]]:
    manifest = os.getenv("SAFE_DATASET_MANIFEST")
    if not manifest:
        return []
    try:
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        videos = data.get("videos", [])
        return [
            {key: video[key] for key in ("path", "filename", "roi", "split", "video_id")}
            for video in videos if isinstance(video, dict)
            and all(key in video for key in ("path", "filename", "roi", "split", "video_id"))
            and isinstance(video["path"], str) and Path(video["path"]).is_file()
            and Path(video["path"]).suffix.lower() == ".mov" and _valid_roi(video["roi"])
        ]
    except (OSError, ValueError, TypeError, AttributeError):
        st.warning("Не удалось прочитать список локальных примеров.")
        return []


@st.cache_data(show_spinner=False, max_entries=4)
def _first_frame(path: str, content: bytes | None = None) -> Image.Image | None:
    """Decode display-oriented pixels; the original MOV is still sent to the API."""
    import cv2

    def decode(filename: str) -> Image.Image | None:
        capture = cv2.VideoCapture(filename)
        try:
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            ok, frame = capture.read()
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None
        finally:
            capture.release()

    try:
        if content is None:
            return decode(path)
        with tempfile.TemporaryDirectory(prefix="safe-preview-") as temporary:
            filename = Path(temporary) / "preview.mov"
            filename.write_bytes(content)
            return decode(str(filename))
    except (OSError, cv2.error, ValueError):
        return None


def _reset_roi_confirmation() -> None:
    st.session_state.roi_confirmed = False


def _safe_upload() -> None:
    st.write("Выберите исходное MOV и проверьте область сейфа на кадре перед анализом.")
    st.caption("Модель распознаёт только открытое/закрытое состояние сейфа. Неопределённый результат требует проверки.")
    source = st.radio("Источник видео", ["Загрузить MOV", "Локальные примеры"], horizontal=True)
    filename, content, path, identity = "", None, None, "none"
    preset = [0.0, 0.0, 1.0, 1.0]
    if source == "Локальные примеры":
        examples = _local_examples()
        if examples:
            index = st.selectbox("Исходное видео", range(len(examples)), format_func=lambda i: examples[i]["filename"])
            example = examples[index]
            filename, path = example["filename"], example["path"]
            identity = path
            preset = example["roi"]
            split = example["split"]
            label = {"train": "Обучающий пример", "val": "Валидационный пример", "test": "Отложенный тест — видео не использовалось для обучения"}.get(split, "Разбиение не указано")
            st.caption(label)
            st.caption("Предустановленная область — калибровка камеры. Аннотации состояний не передаются модели как предсказания.")
        else:
            st.info("Локальные примеры недоступны. Загрузите исходное MOV.")
    else:
        upload = st.file_uploader("Видео в формате .mov", type=["mov"], accept_multiple_files=False)
        if upload is not None:
            filename, content = upload.name, upload.getvalue()
            identity = hashlib.sha256(content).hexdigest()
        st.caption("Для нового видео вручную задайте область сейфа. Начальная рамка охватывает весь кадр и требует изменения или явного подтверждения.")
    if st.session_state.get("roi_source") != identity:
        st.session_state.roi_source = identity
        for key, value in zip(("roi_x", "roi_y", "roi_w", "roi_h"), preset):
            st.session_state[key] = float(value)
        st.session_state.roi_confirmed = False
    st.session_state.selected_video_identity = identity
    st.markdown("**Область сейфа (ROI)**")
    st.caption("Координаты от 0 до 1: X/Y — левый верхний угол, ширина/высота — доли кадра. Рамка должна содержать сейф.")
    columns = st.columns(4)
    roi = []
    for column, key, label in zip(columns, ("roi_x", "roi_y", "roi_w", "roi_h"), ("X", "Y", "Ширина", "Высота")):
        with column:
            roi.append(st.number_input(label, min_value=0.0, max_value=1.0, step=0.01, key=key, on_change=_reset_roi_confirmation))
    valid_roi = _valid_roi(roi)
    if not valid_roi:
        st.error("Область должна иметь положительный размер и полностью находиться в кадре: X + ширина ≤ 1, Y + высота ≤ 1.")
    frame = _first_frame(path or filename, content) if filename else None
    if frame is not None and valid_roi:
        preview = frame.copy()
        x, y, w, h = roi
        draw = ImageDraw.Draw(preview)
        draw.rectangle((int(x * frame.width), int(y * frame.height), min(frame.width - 1, int((x + w) * frame.width)), min(frame.height - 1, int((y + h) * frame.height))), outline="#35ff75", width=max(2, frame.width // 250))
        st.image(preview, caption="Первый кадр с выбранной областью сейфа", width="stretch")
    elif filename:
        st.warning("Не удалось показать первый кадр. Выберите читаемый MOV для визуальной проверки области сейфа.")
    else:
        st.info("Выберите видео, чтобы увидеть кадр и рамку области сейфа.")
    confirmed = st.checkbox("Я проверил(а) рамку: в ней находится нужный сейф", key="roi_confirmed")
    if st.button("Анализировать", type="primary"):
        if frame is None or not valid_roi or not confirmed:
            st.error("Покажите кадр и подтвердите корректную область сейфа перед анализом.")
        else:
            try:
                original = content if content is not None else Path(path).read_bytes()
                _accept_job(submit_video(load_settings(), filename, original, roi=roi))
                st.session_state.job_source_identity = identity
                st.session_state.job_filename = filename
            except (APIClientError, OSError) as exc:
                st.error(str(exc) if isinstance(exc, APIClientError) else "Не удалось прочитать локальное видео.")


def _show_model_report() -> None:
    report_path = os.getenv("SAFE_MODEL_REPORT")
    if not report_path and os.getenv("MODEL_PATH"):
        report_path = str(Path(os.environ["MODEL_PATH"]) / "report.json")
    with st.expander("Проверка модели и ограничения"):
        st.write("Базовая модель обучена на кадрах предоставленных камер. Результаты на этих камерах не гарантируют перенос на новые помещения и ракурсы. Редкие кадровые метки не подтверждают точность временных интервалов событий.")
        if not report_path:
            st.caption("Отчёт об отложенном тесте не настроен.")
            return
        try:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            test = report["splits"]["test"]
            st.write(f"Отложенный тест: {test['samples']} кадров · {test['videos']} видео · {test['groups']} независимых групп.")
            st.write(f"Классы: {test.get('class_counts', {})}")
            coverage = test.get("coverage")
            if isinstance(coverage, (int, float)):
                st.write(f"Доля кадров без отказа от решения: {coverage:.1%}; неопределённых: {test.get('abstained', 'не указано')}.")
            metrics = test.get("per_class", {})
            if isinstance(metrics, dict):
                st.dataframe([{"класс": name, **values} for name, values in metrics.items() if isinstance(values, dict)], hide_index=True)
            st.caption("Precision/recall/F1 и матрица ошибок относятся к бинарным решениям до отказа при низкой уверенности. Это кадровые метрики, не точность обнаружения событий.")
            st.write("Матрица ошибок (строки/столбцы: closed, open):", test.get("confusion_matrix", []))
            for limitation in report.get("limitations", []):
                st.caption(str(limitation))
        except (OSError, ValueError, KeyError, TypeError):
            st.caption("Отчёт модели пока недоступен или имеет неподдерживаемый формат.")


def _accept_job(response: dict[str, Any]) -> None:
    job_id = response.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise APIClientError("API не вернул идентификатор задачи.")
    st.session_state.job_id = job_id
    st.session_state.job_status = response.get("status", "processing")
    st.session_state.job_data = response
    st.session_state.job_error = "Не удалось обработать видео. Проверьте сервис и попробуйте снова."
    st.query_params["job_id"] = job_id


def _job_fragment() -> None:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return

    if os.getenv("ANALYZER_CALLABLE") == "src.vision.safe_classifier:analyze_video":
        selected = st.session_state.get("selected_video_identity", "none")
        analyzed = st.session_state.get("job_source_identity")
        if selected != "none" and selected != analyzed:
            st.warning("Выбранное видео ещё не анализировалось; ниже результат предыдущего запуска.")
        filename = st.session_state.get("job_filename")
        if filename:
            st.markdown(f"**Результат для видео: {filename}**")
        else:
            st.caption("Сохранённый результат отдельного запуска")

    status = st.session_state.get("job_status", "processing")
    if status == "processing":
        st.info("Видео анализируется. Состояние обновляется автоматически.")
        try:
            job = get_job(load_settings(), job_id)
            st.session_state.job_data = job
            new_status = job.get("status")
            if new_status in ("processing", "done", "failed"):
                st.session_state.job_status = new_status
                if new_status == "failed":
                    error = job.get("error")
                    if isinstance(error, str) and error.strip():
                        st.session_state.job_error = error[:500]
            else:
                st.session_state.job_status = "failed"
                st.session_state.job_error = "API вернул неизвестное состояние задачи."
            status = st.session_state.job_status
        except APIClientError as exc:
            # A transient poll failure should not discard the active job.
            _show_demo_warning(bool(st.session_state.get("job_data", {}).get("demo", False)))
            st.warning(str(exc))
            return

    _show_demo_warning(bool(st.session_state.get("job_data", {}).get("demo", False)))
    if status == "done":
        _render_result(st.session_state.get("job_data", {}))
    elif status == "failed":
        st.error(st.session_state.get("job_error", "Не удалось обработать видео."))


def main() -> None:
    # Loads .env before selecting the analyzer-specific interface.
    load_settings()
    requested_job = st.query_params.get("job_id")
    if requested_job and requested_job != st.session_state.get("job_id"):
        st.session_state.job_id = requested_job
        st.session_state.job_status = "processing"
        st.session_state.job_data = {}
        st.session_state.pop("job_source_identity", None)
        st.session_state.pop("job_filename", None)
    st.title("Проверка видео")
    st.write("Загрузите видео .mov, чтобы запустить локальный анализ.")
    st.caption("Видео обрабатывается локальным сервисом и не отправляется облачному AI-сервису.")
    st.info(
        "Результаты требуют проверки человеком. Интерфейс помогает найти фрагменты видео и не выносит окончательное решение."
    )

    safe_mode = os.getenv("ANALYZER_CALLABLE") == "src.vision.safe_classifier:analyze_video"
    if safe_mode:
        _safe_upload()
        _show_model_report()
    else:
        with st.form("video_upload", clear_on_submit=False):
            uploaded_file = st.file_uploader("Видео в формате .mov", type=["mov"], accept_multiple_files=False)
            submitted = st.form_submit_button("Анализировать", type="primary")

        if submitted:
            if uploaded_file is None:
                st.error("Сначала выберите видеофайл .mov.")
            else:
                try:
                    _accept_job(submit_video(load_settings(), uploaded_file.name, uploaded_file.getvalue()))
                except APIClientError as exc:
                    st.error(str(exc))

    if st.session_state.get("job_id"):
        st.divider()
        # Streamlit fragments refresh this area alone, so uploading is never repeated.
        fragment = getattr(st, "fragment", None)
        if fragment is not None:
            fragment(run_every="2s")(_job_fragment)()
        else:
            _job_fragment()
            if st.session_state.get("job_status") == "processing" and st.button("Обновить состояние"):
                st.rerun()


if __name__ == "__main__":
    main()
