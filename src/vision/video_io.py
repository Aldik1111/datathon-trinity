"""Decode full videos with presentation timestamps and MOV display orientation."""
from __future__ import annotations

import math

import av
import cv2
import numpy as np


def apply_rotation(image: np.ndarray, rotation: float) -> np.ndarray:
    """Apply AV's counterclockwise display-matrix angle, without stretching."""
    angle = float(rotation) % 360
    if not math.isfinite(angle):
        raise ValueError('Video has invalid display orientation')
    if min(abs(angle - round(angle / 90) * 90), abs(angle - 360)) > 1:
        raise ValueError('Video display orientation must be a multiple of 90 degrees')
    turns = int(round(angle / 90)) % 4
    return np.ascontiguousarray(np.rot90(image, turns)) if turns else image


def iter_sampled_frames(path, sample_fps=2):
    """Yield relative PTS and display-oriented BGR frames; consume through EOF.

    A late decode/timestamp failure raises even after samples have been yielded,
    so callers must not publish a partial result. No frame-index timing fallback.
    """
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 30:
        raise ValueError('Invalid sampling frequency')
    first_pts, previous, next_sample, count, decoded = None, None, 0., 0, 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('Video contains no video stream')
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            decoded += 1
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Video frame has no presentation timestamp')
            pts = float(frame.pts * frame.time_base)
            if not math.isfinite(pts) or (previous is not None and pts < previous - 1e-6):
                raise ValueError('Video presentation timestamps are invalid')
            previous = pts
            if first_pts is None:
                first_pts = pts
            elapsed = max(0., pts - first_pts)
            if elapsed > 300 + 1e-6:
                raise ValueError('Video exceeds the 300 second recognition limit')
            if elapsed + 1e-6 < next_sample:
                continue
            count += 1
            if count > 1200:
                raise ValueError('Video exceeds the 1200 sample recognition limit')
            rotation = getattr(frame, 'rotation', 0)
            # Legacy files can expose a rotate tag without display side data.
            if not rotation and stream.metadata.get('rotate'):
                rotation = -float(stream.metadata['rotate'])
            image = apply_rotation(frame.to_ndarray(format='bgr24'), rotation)
            yield elapsed, image
            next_sample = (math.floor((elapsed + 1e-6) * sample_fps) + 1) / sample_fps
    if not decoded:
        raise ValueError('Video contains no decodable frames')
