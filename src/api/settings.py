"""Configuration for the HTTP API and its analysis worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    upload_dir: Path
    max_upload_mb: int
    model_path: str | None = None
    analyzer_mode: str = "real"
    analyzer_callable: str = "src.vision.analyzer:analyze_video"
    rules_path: str | None = "configs/rules.json"
    analyzer_timeout_sec: int = 300
    max_pending_jobs: int = 8
    job_ttl_sec: int = 24 * 60 * 60
    db_path: Path | None = None
    scenario_model_path: str | None = None
    # Preserve the historical configured-analyzer route unless an operator opts
    # into one of the experimental analyzers as the default.
    default_analysis_type: str | None = None

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("API_KEY must not be empty")
        if self.max_upload_mb < 1:
            raise ValueError("MAX_UPLOAD_MB must be at least 1")
        if self.analyzer_mode not in {"demo", "real"}:
            raise ValueError("ANALYZER_MODE must be 'demo' or 'real'")
        if self.default_analysis_type not in {None, "jury", "safe_state", "bank_scenarios"}:
            raise ValueError("DEFAULT_ANALYSIS_TYPE must be jury, safe_state, or bank_scenarios")
        if not 1 <= self.analyzer_timeout_sec <= 300:
            raise ValueError("ANALYZER_TIMEOUT_SEC must be between 1 and 300")
        if self.max_pending_jobs < 1:
            raise ValueError("MAX_PENDING_JOBS must be positive")
        if self.job_ttl_sec < 1:
            raise ValueError("JOB_TTL_SEC must be positive")

    @property
    def database_path(self) -> Path:
        return self.db_path or self.upload_dir / "jobs.sqlite3"


def get_settings() -> Settings:
    load_dotenv(override=False)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY must be set in the environment or .env file")
    upload_dir = Path(os.getenv("UPLOAD_DIR", "artifacts/uploads")).expanduser()
    return Settings(
        api_key=api_key,
        upload_dir=upload_dir,
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "100")),
        model_path=os.getenv("MODEL_PATH") or None,
        scenario_model_path=os.getenv("SCENARIO_MODEL_PATH") or None,
        default_analysis_type=(os.getenv("DEFAULT_ANALYSIS_TYPE", "").strip().lower() or None),
        analyzer_mode=os.getenv("ANALYZER_MODE", "real").lower(),
        analyzer_callable=os.getenv(
            "ANALYZER_CALLABLE", "src.vision.analyzer:analyze_video"
        ),
        rules_path=os.getenv("RULES_PATH", "configs/rules.json") or None,
        analyzer_timeout_sec=int(os.getenv("ANALYZER_TIMEOUT_SEC", "300")),
        max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", "2")),
        job_ttl_sec=int(os.getenv("JOB_TTL_SEC", str(24 * 60 * 60))),
        db_path=Path(os.environ["API_DB_PATH"]).expanduser()
        if os.getenv("API_DB_PATH")
        else None,
    )
