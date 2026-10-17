"""Deterministic model stand-ins used by spawned API integration tests."""

from __future__ import annotations

import time


def no_violation(video_path: str) -> dict:
    """Return a valid empty result without classifying the input."""
    return {"verdict": "no_violation", "events": []}


def wait_then_no_violation(video_path: str) -> dict:
    """Stay busy long enough to test API responsiveness and timeout handling."""
    time.sleep(5)
    return no_violation(video_path)
