"""Experimental scenario controls and comparison, without label-assisted inference."""

from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

import streamlit as st

from src.ui import client


MODE_TITLES = {"bank_scenarios": "Сценарии из таблицы (эксперимент)", "safe_state": "Состояние сейфа"}
DECISIONS = {"candidate": "Кандидат для проверки", "not_flagged": "Ниже порога", "unsupported": "Нет поддержки"}
EVALUATIONS = {"out_of_fold": "Вне обучающей группы (OOF)", "in_sample": "Финальная модель · обучающие видео"}


def local_examples() -> list[dict[str, str]]:
    """Read only the fields necessary to choose the original media for upload."""
    manifest = os.getenv("SCENARIO_DATASET_MANIFEST")
    if not manifest:
        return []
    try:
        rows = json.loads(Path(manifest).read_text(encoding="utf-8")).get("rows", [])
        examples = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            media = row.get("original_media")
            if not isinstance(media, dict):
                continue
            path, filename, row_id = media.get("path"), row.get("filename"), row.get("row_id")
            if (isinstance(path, str) and Path(path).is_file() and Path(path).suffix.lower() == ".mov"
                    and isinstance(filename, str) and isinstance(row_id, str)):
                examples.append({"path": path, "filename": filename, "row_id": row_id})
        return examples
    except (OSError, ValueError, TypeError, AttributeError):
        st.warning("Не удалось прочитать список локальных видео для сценариев.")
        return []


def upload(accept_job: Callable[[dict[str, Any]], None]) -> None:
    st.warning("Эксперимент со слабыми метками. Поддержка типов зависит от числа независимых примеров и показана в результате. Модель предлагает кандидатов; причина действия и нарушение требуют проверки человеком.")
    quality_notice()
    examples = local_examples()
    options = ["Локальные примеры", "Загрузить MOV"] if examples else ["Загрузить MOV", "Локальные примеры"]
    source = st.radio("Источник видео", options, horizontal=True, key="scenario_source")
    filename, content, path, identity = "", None, None, "none"
    if source == "Локальные примеры":
        if examples:
            st.caption(f"Доступно исходных видео: {len(examples)}. Разметка не передаётся модели при анализе.")
            index = st.selectbox("Исходное видео", range(len(examples)), key="scenario_example",
                                 format_func=lambda i: examples[i]["filename"])
            example = examples[index]
            filename, path = example["filename"], example["path"]
            identity = f"{example['row_id']}:{path}:{filename}"
        else:
            st.info("Локальные примеры недоступны. Загрузите исходное MOV.")
    else:
        video = st.file_uploader("Видео в формате .mov", type=["mov"], accept_multiple_files=False, key="scenario_upload")
        if video is not None:
            filename, content = video.name, video.getvalue()
            identity = f"{filename}:{hashlib.sha256(content).hexdigest()}"
    st.session_state.selected_video_identity = identity
    st.caption("Анализируется весь кадр. Временные окна показывают пики оценок, а не подтверждённые начало и конец действия.")
    if st.button("Анализировать", type="primary", key="scenario_analyze"):
        if not filename:
            st.error("Сначала выберите видеофайл .mov.")
        else:
            try:
                original = content if content is not None else Path(path).read_bytes()
                accept_job(client.submit_video(client.load_settings(), filename, original, analysis_type="bank_scenarios"))
                st.session_state.job_source_identity = identity
                st.session_state.job_filename = filename
                st.session_state.job_analysis_type = "bank_scenarios"
            except (client.APIClientError, OSError) as exc:
                st.error(str(exc) if isinstance(exc, client.APIClientError) else "Не удалось прочитать локальное видео.")


def quality_notice() -> None:
    path = os.getenv("SCENARIO_COMPARISON_REPORT")
    if not path:
        return
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        grouped = report["out_of_fold"]
        micro, binary = grouped["micro"], grouped["overall_binary"]
        values = [micro["tp"], micro["scored_positive"], micro["unsupported"],
                  binary.get("normal_alarm_rows", binary["fp"]),
                  binary.get("normal_rows", binary["scored_negative"])]
        if any(type(value) is not int or value < 0 for value in values):
            return
    except (OSError, ValueError, KeyError, TypeError):
        return
    tp, positives, unsupported, false_alarms, normals = values
    st.error(f"Групповая проверка на известном наборе: найдено {tp} из {positives} оцениваемых положительных меток; "
             f"ложные тревоги на {false_alarms} из {normals} нормальных видео. "
             f"Без поддержки: {unsupported} меток. Результаты требуют ручной проверки.")
    st.caption("OOF исключает целую группу дублей из обучения. Это не проверка на новых независимых съёмках; неподдерживаемые типы не считаются отсутствующими.")


def _class_table(classes: Any) -> list[dict[str, Any]]:
    if not isinstance(classes, list):
        return []
    return [{"Тип": item.get("title", item.get("type")), "Оценка модели": item.get("score"),
             "Порог": item.get("threshold"), "Решение": DECISIONS.get(item.get("decision"), "Неизвестно"),
             "Положительных групп": item.get("positive_groups"), "Отрицательных групп": item.get("negative_groups"),
             "Поддержка": "Нет поддержки" if item.get("decision") == "unsupported" else
                          ("1 пример · редкий тип" if item.get("positive_groups") == 1 else "Несколько примеров"),
             "Пик": client.format_timecode(item.get("peak_sec")) if item.get("peak_sec") is not None else "—"}
            for item in classes if isinstance(item, dict)]


def render_result(job: dict[str, Any]) -> None:
    classes = job.get("class_results", [])
    titles = {item.get("type"): item.get("title", item.get("type")) for item in classes if isinstance(item, dict)} if isinstance(classes, list) else {}
    if job.get("verdict") == "review_required":
        st.warning("Найдены кандидаты сценариев для ручной проверки")
    elif job.get("verdict") == "no_candidate":
        st.info("Ни одна оценка не превысила порог. Это не подтверждает отсутствие нарушений.")
    else:
        st.error("API вернул неизвестный итог сценарного анализа.")
    st.caption(f"Модель: {job.get('model_version', 'не указана')} · Признаки: {job.get('feature_version', 'не указаны')}")
    summary = job.get("summary", {})
    if isinstance(summary, dict):
        st.write(f"Проверено кадров: {summary.get('sample_count', '—')} · Частота выборки: {summary.get('sample_fps', '—')} кадр/с")
        if summary.get("events_truncated"):
            st.warning("Достигнут лимит событий; показаны не все окна кандидатов.")
    st.markdown("**Оценки всех сценариев**")
    st.dataframe(_class_table(classes), hide_index=True, width="stretch")
    st.caption("Оценка модели — не калиброванная вероятность. «Ниже порога» не означает, что действие отсутствует; «Нет поддержки» означает, что модель не может оценить этот тип.")
    st.info("Результат не подтверждает причину действия: личность клиента, содержание отчёта, сетевое соединение, принадлежность денег и работу замка нельзя установить только по похожим кадрам.")
    st.caption("Нормативная квалификация кандидатов не подтверждена: видимое действие и нарушение регламента проверяются отдельно.")
    events = job.get("violations", [])
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        with st.container(border=True):
            start = client.format_timecode(event.get("start_sec", event.get("timecode_sec")))
            end = client.format_timecode(event.get("end_sec", event.get("timecode_sec")))
            st.markdown(f"**Кандидат: {titles.get(event.get('type'), event.get('type', 'Сценарий'))} · {start}–{end}**")
            st.caption("Окно высокой оценки модели; время действия требует ручной проверки.")
            frames = event.get("evidence_frames", [])
            valid = []
            for frame in frames if isinstance(frames, list) else []:
                if isinstance(frame, dict):
                    pixels = client.decode_evidence_image(frame.get("image_base64"))
                    if pixels:
                        valid.append((frame.get("timestamp_sec"), pixels))
            if valid:
                # Three slots keep a single portrait frame from occupying the whole page.
                columns = st.columns(3)
                for index, (timestamp, pixels) in enumerate(valid[:3]):
                    with columns[index]:
                        st.image(pixels, caption=f"Кадр · {client.format_timecode(timestamp)}", width="stretch")
            elif frames:
                st.caption("Не удалось прочитать приложенные кадры.")
    limitations = job.get("limitations")
    if isinstance(limitations, list) and limitations:
        with st.expander("Ограничения модели"):
            for limitation in limitations:
                st.write(str(limitation))
    st.download_button("Скачать результат JSON", json.dumps(job, ensure_ascii=False, indent=2),
                       file_name=f"scenario-result-{job.get('job_id', 'analysis')}.json", mime="application/json")


def _joined(items: Any, titles: dict[str, str] | None = None) -> str:
    titles = titles or {}
    return ", ".join(str(titles.get(item, item)) for item in items) if isinstance(items, list) and items else "—"


def _comparison_rows(rows: list[dict[str, Any]], mode: str, titles: dict[str, str]) -> list[dict[str, Any]]:
    table = []
    for row in rows:
        prediction = row.get(mode, {})
        if not isinstance(prediction, dict):
            prediction = {}
        table.append({"Строка": row.get("row_id"), "Видео": row.get("filename"),
                      "Метки таблицы": "Без нарушения по таблице" if row.get("table_violation") is False else _joined(row.get("expected_types"), titles),
                      "Кандидаты модели": _joined(prediction.get("candidate_types"), titles),
                      "Без поддержки": _joined(prediction.get("unsupported_types"), titles),
                      "Проверка источника": {"review": "На проверке", "needs_review": "На проверке", "source_review": "На проверке", "done": "Размечено"}.get(row.get("annotation_status"), "Не указана"),
                      "Конфликт дубля": "Да" if row.get("duplicate_conflict") else "Нет"})
    return table


def _comparison_csv(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["evaluation", "row_id", "filename", "expected_types", "candidate_types", "unsupported_types", "annotation_status", "duplicate_conflict"])
    writer.writeheader()
    for mode in EVALUATIONS:
        for row in rows:
            prediction = row.get(mode, {})
            writer.writerow({"evaluation": mode, "row_id": row.get("row_id"), "filename": row.get("filename"),
                             "expected_types": _joined(row.get("expected_types")),
                             "candidate_types": _joined(prediction.get("candidate_types")),
                             "unsupported_types": _joined(prediction.get("unsupported_types")),
                             "annotation_status": row.get("annotation_status"), "duplicate_conflict": bool(row.get("duplicate_conflict"))})
    return "\ufeff" + output.getvalue()


def comparison_report() -> None:
    with st.expander("Сравнение со всей таблицей · обучение и независимая проверка"):
        path = os.getenv("SCENARIO_COMPARISON_REPORT")
        if not path:
            st.caption("Отчёт сравнения ещё не настроен.")
            return
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            rows = report["rows"]
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("Invalid comparison rows")
        except (OSError, ValueError, KeyError, TypeError):
            st.warning("Отчёт сравнения недоступен или имеет неподдерживаемый формат.")
            return
        summary = report.get("summary", {})
        summary = summary if isinstance(summary, dict) else {}
        titles = {item.get("type"): item.get("title", item.get("type")) for item in report.get("classes", []) if isinstance(item, dict)}
        st.write(f"Всего строк: {len(rows)} · Независимых групп: {summary.get('groups', '—')} · Типов: {summary.get('types', '—')}")
        reviews = sum(row.get("annotation_status") in ("review", "needs_review", "source_review") for row in rows)
        conflicts = sum(bool(row.get("duplicate_conflict")) for row in rows)
        st.caption(f"На проверке источника: {reviews} · Конфликты дублей: {conflicts}")
        st.caption("Метки таблицы — слабые ориентиры. Пропущенный тип в положительном видео не считается отрицательной меткой; редкие временные отметки не являются точной разметкой интервалов.")
        mode = st.radio("Результаты сравнения", list(EVALUATIONS), format_func=EVALUATIONS.get,
                        key="scenario_comparison_mode", horizontal=True)
        if mode == "out_of_fold":
            st.info("OOF: каждое видео проверено моделью, которая не обучалась на его группе и её дублях. Тип без положительного примера в обучении отмечен как неподдерживаемый.")
        else:
            st.warning("Resubstitution: финальная модель проверена на видео, использованных для обучения. Это не независимая оценка качества.")
        unsupported = sum(len(row.get(mode, {}).get("unsupported_types", [])) for row in rows)
        st.write(f"Неподдерживаемых пар видео/тип: {unsupported}")
        metrics = report.get(mode, {})
        if isinstance(metrics, dict):
            micro = metrics.get("micro")
            if isinstance(micro, dict) and micro:
                st.write("Сводные метрики по известным меткам:", micro)
            per_type = metrics.get("per_type")
            if isinstance(per_type, dict) and per_type:
                st.dataframe([{"Тип": titles.get(kind, kind), **values} for kind, values in per_type.items() if isinstance(values, dict)], hide_index=True, width="stretch")
            elif isinstance(per_type, list) and per_type:
                st.dataframe(per_type, hide_index=True, width="stretch")
        row_ids = [str(row.get("row_id", index)) for index, row in enumerate(rows)]
        chosen = st.selectbox("Подробности видео", ["all", *row_ids], key="scenario_comparison_row",
                              format_func=lambda value: "Все видео" if value == "all" else value)
        selected = rows if chosen == "all" else [row for row, row_id in zip(rows, row_ids) if row_id == chosen]
        st.dataframe(_comparison_rows(selected, mode, titles), hide_index=True, width="stretch")
        if chosen != "all" and selected:
            row = selected[0]
            st.write(f"Видео: {row.get('filename')}")
            job_id = row.get("final_job_id")
            if isinstance(job_id, str) and job_id:
                base_url = os.getenv("UI_URL", "http://127.0.0.1:8501").rstrip("/")
                st.link_button("Открыть результат полного прогона", f"{base_url}/?{urlencode({'job_id': job_id})}")
            if row.get("user_description"):
                st.caption(f"Описание в исходной таблице: {row['user_description']}")
            classes = row.get(mode, {}).get("class_results", [])
            if classes:
                st.dataframe(_class_table(classes), hide_index=True, width="stretch")
        left, right = st.columns(2)
        with left:
            st.download_button("Скачать сравнение JSON", json.dumps(report, ensure_ascii=False, indent=2),
                               file_name="scenario-comparison.json", mime="application/json", key="scenario_comparison_json")
        with right:
            st.download_button("Скачать сравнение CSV", _comparison_csv(rows),
                               file_name="scenario-comparison.csv", mime="text/csv", key="scenario_comparison_csv")
