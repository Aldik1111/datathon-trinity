"""Small HTTP client for the local video analysis API."""

from __future__ import annotations

import base64
import binascii
from decimal import Decimal, InvalidOperation
from io import BytesIO
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
import requests
from src.runtime_config import read_api_key


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_URL = "http://127.0.0.1:8000"
UPLOAD_TIMEOUT = (5, 120)
POLL_TIMEOUT = (5, 15)


@dataclass(frozen=True)
class APISettings:
    base_url: str
    api_key: str


def load_settings() -> APISettings:
    """Load local .env settings without ever showing or logging the key."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        # Environment variables still work when python-dotenv is unavailable.
        pass

    try:
        api_key = read_api_key()
    except ValueError:
        raise APIClientError("Не удалось прочитать API-ключ. Проверьте настройки сервиса.") from None
    return APISettings(
        base_url=os.getenv("API_URL", DEFAULT_API_URL).rstrip("/"),
        api_key=api_key,
    )


class APIClientError(RuntimeError):
    """A safe, user-displayable API error that never contains request headers."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _headers(settings: APISettings) -> dict[str, str]:
    if not settings.api_key:
        raise APIClientError(
            "Не задан API_KEY. Добавьте ключ в локальный .env рядом с проектом и перезапустите приложение."
        )
    return {"X-API-Key": settings.api_key}


def _response_error(response: requests.Response, api_key: str = "") -> APIClientError:
    message = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("message")
            if isinstance(detail, str):
                message = detail[:500]
                if api_key:
                    message = message.replace(api_key, "[скрыто]")
    except (ValueError, requests.JSONDecodeError):
        pass
    if not message:
        message = "Сервис вернул ошибку. Проверьте, что API запущен, и повторите попытку."
    return APIClientError(f"Ошибка API ({response.status_code}): {message}", response.status_code)


def _json_object(response: requests.Response, api_key: str = "") -> dict[str, Any]:
    if not response.ok:
        raise _response_error(response, api_key)
    try:
        data = response.json()
    except (ValueError, requests.JSONDecodeError) as exc:
        raise APIClientError("API вернул некорректный ответ.") from exc
    if not isinstance(data, dict):
        raise APIClientError("API вернул ответ неверного формата.")
    return data


def submit_video(settings: APISettings, filename: str, video_bytes: bytes) -> dict[str, Any]:
    """Submit one .mov file and return the API's job response."""
    if Path(filename).suffix.lower() != ".mov":
        raise APIClientError("Поддерживаются видеофайлы в формате .mov.")
    if not video_bytes:
        raise APIClientError("Выбранный файл пуст.")
    try:
        response = requests.post(
            f"{settings.base_url}/api/v1/analyze",
            headers=_headers(settings),
            files={"file": (Path(filename).name, video_bytes, "video/quicktime")},
            timeout=UPLOAD_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise APIClientError("Сервис не ответил за отведённое время. Попробуйте ещё раз.") from exc
    except requests.RequestException as exc:
        raise APIClientError("Не удалось связаться с API. Проверьте, что локальный сервис запущен.") from exc
    return _json_object(response, settings.api_key)


def get_job(settings: APISettings, job_id: str) -> dict[str, Any]:
    """Fetch the latest state and result for a previously created job."""
    if not job_id or "/" in job_id or "\\" in job_id:
        raise APIClientError("API вернул некорректный идентификатор задачи.")
    try:
        response = requests.get(
            f"{settings.base_url}/api/v1/jobs/{job_id}",
            headers=_headers(settings),
            timeout=POLL_TIMEOUT,
        )
    except requests.Timeout as exc:
        raise APIClientError("Истекло время ожидания ответа API. Обновление повторится автоматически.") from exc
    except requests.RequestException as exc:
        raise APIClientError("Не удалось получить состояние задачи. Проверьте, что локальный API запущен.") from exc
    return _json_object(response, settings.api_key)


def decode_evidence_image(value: object) -> bytes | None:
    """Decode and validate a base64 JPEG evidence frame; malformed frames are skipped."""
    if not isinstance(value, str) or not value:
        return None
    # Accept an optional data URL prefix while keeping the API's raw-base64 form.
    if value.startswith("data:") and "," in value:
        value = value.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(value, validate=True)
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format != "JPEG":
                return None
            image.verify()
        return image_bytes
    except (binascii.Error, ValueError, OSError, Image.DecompressionBombError):
        return None


def format_timecode(value: object) -> str:
    """Format seconds as MM:SS while preserving fractional frame timestamps."""
    try:
        total = max(Decimal(0), Decimal(str(value)))
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return "время не указано"
    if not total.is_finite():
        return "время не указано"
    hours = int(total // 3600)
    remainder = total - hours * 3600
    minutes = int(remainder // 60)
    seconds = remainder - minutes * 60
    whole_seconds = int(seconds)
    fractional = str(seconds.normalize()).partition(".")[2].rstrip("0")
    second_text = f"{whole_seconds:02d}" + (f".{fractional}" if fractional else "")
    if hours:
        return f"{hours:02d}:{minutes:02d}:{second_text}"
    return f"{minutes:02d}:{second_text}"
