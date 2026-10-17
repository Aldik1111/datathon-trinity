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


def noisy_failure(video_path: str) -> dict:
    """Synthetic canary must never reach public responses or server logs."""
    import os
    import sys
    marker = "SYNTHETIC_PRIVATE_CANARY"
    print(marker, flush=True)
    print(marker, file=sys.stderr, flush=True)
    os.write(2, marker.encode())
    raise RuntimeError(marker)
