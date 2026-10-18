"""Scoped safe open/closed observations; no identity or regulatory inference."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2

from .safe_model import load_model, predict_crop
from .video_io import iter_sampled_frames


def read_roi(video_path):
    try:
        roi = json.loads(Path(str(video_path) + '.roi.json').read_text(encoding='utf-8'))['roi']
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError()
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in roi):
            raise ValueError()
        x, y, width, height = roi
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1 + 1e-9 or y + height > 1 + 1e-9:
            raise ValueError()
        return roi
    except (OSError, KeyError, ValueError, TypeError):
        raise ValueError('A valid normalized safe ROI sidecar is required') from None


def _box(image, roi):
    height, width = image.shape[:2]
    x, y, w, h = roi
    return (min(width - 1, int(x * width)), min(height - 1, int(y * height)),
            min(width, math.ceil((x + w) * width)), min(height, math.ceil((y + h) * height)))


def _evidence(image, box, timestamp, state):
    marked = image.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(marked, (x1, y1), (x2 - 1, y2 - 1), (0, 210, 255), 2)
    cv2.putText(marked, f'SAFE ROI: {state}', (max(0, x1), max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 210, 255), 1)
    height, width = marked.shape[:2]
    if max(height, width) > 1280:
        marked = cv2.resize(marked, (round(width * 1280 / max(height, width)), round(height * 1280 / max(height, width))))
    ok, jpeg = cv2.imencode('.jpg', marked, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('Evidence JPEG encoding failed')
    return dict(timestamp_sec=float(timestamp), jpeg_bytes=jpeg.tobytes())


def analyze_video(video_path: str) -> dict:
    roi = read_roi(video_path)
    model_path = os.environ.get('MODEL_PATH')
    if not model_path:
        raise ValueError('MODEL_PATH must point to a trained safe model directory')
    model = load_model(model_path)
    observations, events = [], []
    counts = dict(open=0, closed=0, unknown=0)
    current, event_count = None, 0
    for timestamp, image in iter_sampled_frames(video_path):
        box = _box(image, roi)
        x1, y1, x2, y2 = box
        prediction = predict_crop(model, image[y1:y2, x1:x2])
        state, confidence = prediction['state'], prediction['confidence']
        observations.append(dict(timecode_sec=float(timestamp), **prediction))
        counts[state] += 1
        if current is not None and current['_state'] != state:
            current = None
        if state == 'closed':
            continue
        if current is None:
            event_count += 1
            current = dict(event_type='safe_open' if state == 'open' else 'safe_uncertain',
                           timecode_sec=float(timestamp), start_sec=float(timestamp), end_sec=float(timestamp),
                           confidence=float(confidence),
                           description=('Selected safe region appears open in sampled frames; human review required.'
                                        if state == 'open' else 'Safe state is uncertain in sampled frames; human review required.'),
                           evidence_frames=[], _state=state)
            if len(events) < 100:
                events.append(current)
        current['end_sec'] = float(timestamp)
        current['confidence'] = min(current['confidence'], float(confidence))
        if event_count <= 100:
            evidence = _evidence(image, box, timestamp, state)
            if len(current['evidence_frames']) < 3:
                current['evidence_frames'].append(evidence)
            else:
                current['evidence_frames'][-1] = evidence
    if not observations:
        raise ValueError('Video contains no sampled frames')
    for event in events:
        event.pop('_state')
    return dict(verdict='review_required' if counts['open'] or counts['unknown'] else 'no_event',
                events=events, analysis_scope='safe_state', model_version=model.metadata['model_version'],
                summary=dict(sample_count=len(observations), state_counts=counts, sample_fps=2,
                             events_truncated=event_count > 100, score_semantics='uncalibrated_model_score',
                             scope_note='Safe state only. No event means no confident open or uncertain sampled state; it does not establish absence of violations.'),
                observations=observations)
