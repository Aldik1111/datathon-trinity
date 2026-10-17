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


def multiple_events(video_path: str):
    """Synthetic adapter fixture, never a recognition model."""
    import cv2
    from src.vision.contracts import AnalysisResult, Event, EvidenceFrame
    capture = cv2.VideoCapture(video_path)
    events = []
    try:
        for timestamp in (0.0, 0.4):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Synthetic fixture unreadable")
            ok, encoded = cv2.imencode(".jpg", frame)
            assert ok
            events.append(Event("synthetic-fixture", timestamp, "Artificial integration test", [EvidenceFrame(timestamp, encoded.tobytes())]))
    finally:
        capture.release()
    return AnalysisResult("violation", events)


def noisy_failure(video_path: str):
    import os
    print("SYNTHETIC_PRIVATE_MARKER", flush=True)
    os.write(2, b"SYNTHETIC_PRIVATE_MARKER\n")
    raise RuntimeError("SYNTHETIC_PRIVATE_MARKER")
