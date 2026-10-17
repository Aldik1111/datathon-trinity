"""Secret loading shared by the API and server-side UI."""
import os
from pathlib import Path


def read_api_key() -> str:
    value = os.getenv("API_KEY", "").strip()
    secret_file = os.getenv("API_KEY_FILE", "").strip()
    if value and secret_file:
        raise ValueError("Configure either API_KEY or API_KEY_FILE, not both")
    if secret_file:
        try:
            value = Path(secret_file).read_text(encoding="utf-8").strip()
        except OSError:
            raise ValueError("API_KEY_FILE cannot be read") from None
    if value and (not value.isascii() or any(char.isspace() for char in value)):
        raise ValueError("API key must be ASCII without whitespace")
    return value
