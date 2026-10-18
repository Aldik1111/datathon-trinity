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


def iter_sampled_frames(path, sample_fps=2, *, coverage=None):
    """Yield relative PTS and display-oriented BGR frames; consume through EOF.

    A late decode/timestamp failure raises even after samples have been yielded,
    so callers must not publish a partial result. No frame-index timing fallback.
    """
    if not math.isfinite(sample_fps) or not 0 < sample_fps <= 30:
        raise ValueError('Invalid sampling frequency')
    if coverage is not None:
        coverage.clear()
        coverage['full_decode'] = False
    first_pts, previous, next_sample, count, decoded = None, None, 0., 0, 0
    max_gap, last_elapsed, last_interval, samples = 0., 0., 0., []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('Video contains no video stream')
        stream = container.streams.video[0]
        declared_frames = stream.frames
        for frame in container.decode(stream):
            decoded += 1
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Video frame has no presentation timestamp')
            pts = float(frame.pts * frame.time_base)
            if not math.isfinite(pts) or (previous is not None and pts < previous - 1e-6):
                raise ValueError('Video presentation timestamps are invalid')
            if previous is not None:
                max_gap = max(max_gap, pts - previous)
            previous = pts
            if first_pts is None:
                first_pts = pts
            elapsed = max(0., pts - first_pts)
            last_elapsed = elapsed
            frame_duration = getattr(frame, 'duration', None)
            last_interval = float(frame_duration * frame.time_base) if frame_duration else 0.
            if not math.isfinite(last_interval) or last_interval <= 0:
                rate = stream.average_rate
                last_interval = float(1 / rate) if rate and rate > 0 else 0.
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
            samples.append(float(elapsed))
            yield elapsed, image
            next_sample = (math.floor((elapsed + 1e-6) * sample_fps) + 1) / sample_fps
    if not decoded:
        raise ValueError('Video contains no decodable frames')
    if declared_frames and declared_frames > 0 and decoded < declared_frames:
        raise ValueError('Video ended before its declared frame count')
    if coverage is not None:
        duration = last_elapsed + last_interval
        if not math.isfinite(duration) or duration <= last_elapsed:
            raise ValueError('Video has no usable end timestamp')
        # Set full_decode only after EOF and integrity checks. A consumer that
        # stops iteration early never receives a successful coverage assertion.
        coverage.update(full_decode=True, decoded_frame_count=decoded,
                        sample_count=count, sample_fps=float(sample_fps),
                        first_sample_sec=samples[0], last_sample_sec=samples[-1],
                        last_frame_sec=float(last_elapsed), duration_sec=float(duration),
                        max_frame_gap_sec=float(max_gap), sample_timestamps_sec=samples)
