"""Frozen image features and short temporal context, computed from pixels only.

MobileNetV2 comes from the official ONNX Model Zoo (Apache-2.0). The frozen
ImageNet backbone does not itself recognize banking actions. Supervised heads
learn weak scenario labels separately. No annotation or media-name lookup.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .video_io import iter_sampled_frames

FEATURE_VERSION = 'mobilenetv2-gap-temporal-v1'
BACKBONE_SHA256 = 'c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5'
BACKBONE_URL = 'https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx'
SAMPLE_FPS = 1.0
FEATURE_DIM = 768
_MEAN = np.array([.485, .456, .406], dtype=np.float32)
_STD = np.array([.229, .224, .225], dtype=np.float32)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 2:
        raise ValueError('Expected a nonempty BGR video frame')
    height, width = frame.shape[:2]
    factor = 224 / max(height, width)
    new_width, new_height = max(1, round(width * factor)), max(1, round(height * factor))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    # Padding equals the ImageNet mean and therefore becomes zero after normalization.
    canvas = np.empty((224, 224, 3), dtype=np.float32)
    canvas[:] = _MEAN
    x, y = (224 - new_width) // 2, (224 - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = resized
    return np.ascontiguousarray(((canvas - _MEAN) / _STD).transpose(2, 0, 1)[None])


class FrameEncoder:
    def __init__(self, backbone_path: str | Path):
        path = Path(backbone_path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != BACKBONE_SHA256:
            raise ValueError('Unexpected scenario backbone checksum')
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layer = 'onnx_node!GlobalAveragePool_97'
        if self.layer not in self.net.getLayerNames():
            raise ValueError('Scenario backbone lacks the expected pooled features')
        # Fixed data-independent projection reduces fitting cost on a small corpus.
        self.projection = np.random.default_rng(1729).normal(0, 1 / 16, (1280, 256)).astype(np.float32)

    def encode(self, frame: np.ndarray) -> np.ndarray:
        self.net.setInput(preprocess_frame(frame))
        pooled = self.net.forward(self.layer).reshape(-1)
        if pooled.shape != (1280,) or not np.isfinite(pooled).all():
            raise ValueError('Backbone returned invalid features')
        return np.asarray(pooled @ self.projection, dtype=np.float32)


def temporal_features(timestamps: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    times = np.asarray(timestamps, dtype=np.float64)
    vectors = np.asarray(embeddings, dtype=np.float32)
    if (times.ndim != 1 or not len(times) or vectors.ndim != 2 or len(times) != len(vectors)
            or not np.isfinite(times).all() or not np.isfinite(vectors).all()
            or (len(times) > 1 and np.any(np.diff(times) <= 0))):
        raise ValueError('Invalid ordered temporal features')
    output = []
    for index, timestamp in enumerate(times):
        context = np.abs(times - timestamp) <= 2 + 1e-6
        before = context & (times < timestamp)
        after = context & (times > timestamp)
        previous = vectors[before].mean(axis=0) if before.any() else vectors[index]
        following = vectors[after].mean(axis=0) if after.any() else vectors[index]
        output.append(np.concatenate([vectors[index], vectors[context].mean(axis=0), following - previous]))
    return np.asarray(output, dtype=np.float32)


def extract_video_features(path, backbone_path, *, encoder=None) -> tuple[np.ndarray, np.ndarray]:
    encoder = encoder if encoder is not None else FrameEncoder(backbone_path)
    times, vectors = [], []
    for timestamp, frame in iter_sampled_frames(path, sample_fps=SAMPLE_FPS):
        times.append(timestamp)
        vectors.append(encoder.encode(frame))
    timestamps = np.asarray(times, dtype=np.float64)
    features = temporal_features(timestamps, np.asarray(vectors))
    if features.shape[1] != FEATURE_DIM:
        raise ValueError('Unexpected scenario feature dimensions')
    return timestamps, features
