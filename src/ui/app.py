"""Streamlit interface for the local video analysis API.

Run from the repository root with:
    streamlit run src/ui/app.py
"""

from __future__ import annotations

from typing import Any

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
    if isinstance(rule, dict) and event.get('rule_status') == 'verified':
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
        if rule.get('checked_at'):
            st.caption(f"Источник проверен: {rule['checked_at']}")
        if rule.get('limitations'):
            st.caption(f"Ограничения: {rule['limitations']}")
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
    verdict = job.get("verdict")
    violations = job.get("violations", [])

    if verdict == "violation":
        if job.get('demo'):
            st.info('Тестовый результат интерфейса: ниже показано искусственное событие, а не обнаруженное нарушение.')
        else:
            st.error("Обнаружены признаки возможных нарушений.")
        if not isinstance(violations, list) or not violations:
            st.info("Сервис не приложил описание событий.")
        else:
            for index, event in enumerate(violations, start=1):
                with st.container(border=True):
                    _render_violation(event, index)
    elif verdict == "no_violation":
        if job.get('demo'):
            st.info('Тестовый отрицательный ответ; содержимое видео не классифицировалось.')
        else:
            st.success("Нарушение не обнаружено.")
    else:
        st.error("API вернул неизвестный итоговый статус.")

    st.info(
        "Результат автоматического анализа является предварительным. Все события и кадры должен проверить человек."
    )


def _job_fragment() -> None:
    job_id = st.session_state.get("job_id")
    if not job_id:
        return

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
    st.title("Проверка видео")
    st.write("Загрузите видео .mov, чтобы запустить локальный анализ.")
    st.caption("Видео обрабатывается локальным сервисом и не отправляется облачному AI-сервису.")
    st.info(
        "Результаты требуют проверки человеком. Интерфейс помогает найти фрагменты видео и не выносит окончательное решение."
    )

    with st.form("video_upload", clear_on_submit=False):
        uploaded_file = st.file_uploader("Видео в формате .mov", type=["mov"], accept_multiple_files=False)
        submitted = st.form_submit_button("Анализировать", type="primary")

    if submitted:
        if uploaded_file is None:
            st.error("Сначала выберите видеофайл .mov.")
        else:
            try:
                with st.spinner('Видео загружается в локальный сервис…'):
                    response = submit_video(load_settings(), uploaded_file.name, uploaded_file.getvalue())
                job_id = response.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    raise APIClientError("API не вернул идентификатор задачи.")
                st.session_state.job_id = job_id
                st.session_state.job_status = response.get("status", "processing")
                st.session_state.job_data = response
                st.session_state.job_error = "Не удалось обработать видео. Проверьте сервис и попробуйте снова."
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
