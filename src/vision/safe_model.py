"""Small CPU safe-state classifier. Artifacts contain numbers, never executable pickle.

Scores are logistic outputs from a sparse, same-camera dataset; they are not
calibrated probabilities or evidence of a regulatory violation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

FEATURE_VERSION = 'hog64-spatial-bgr-hsv-v1'


def valid_crop(image) -> bool:
    return (isinstance(image, np.ndarray) and image.dtype == np.uint8
            and image.ndim == 3 and image.shape[2] == 3
            and min(image.shape[:2]) >= 16
            and .15 <= image.shape[1] / image.shape[0] <= 6)


def features(image_bgr: np.ndarray) -> np.ndarray:
    if not valid_crop(image_bgr):
        raise ValueError('Invalid safe crop: expected a uint8 BGR image of at least 16 pixels')
    image = cv2.resize(image_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9).compute(gray).ravel()
    # Coarse spatial color preserves the door/background arrangement.
    blocks = image.astype(np.float64).reshape(4, 16, 4, 16, 3)
    color = np.concatenate([blocks.mean(axis=(1, 3)).ravel(), blocks.std(axis=(1, 3)).ravel()]) / 255.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = np.concatenate([np.histogram(hsv[:, :, channel], bins=8, range=(0, upper))[0]
                           for channel, upper in ((0, 180), (1, 256), (2, 256))]) / 4096.
    return np.concatenate([hog, color, hist]).astype(np.float64)


@dataclass
class SafeModel:
    coef: np.ndarray
    intercept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict


def load_model(model_dir) -> SafeModel:
    directory = Path(model_dir)
    try:
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        artifact = (directory / 'model.npz').read_bytes()
        if metadata['artifact_sha256'] != hashlib.sha256(artifact).hexdigest():
            raise ValueError('Artifact checksum mismatch')
        if metadata['feature_version'] != FEATURE_VERSION or metadata['schema_version'] != 1:
            raise ValueError('Unsupported model version')
        threshold = metadata['threshold']
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not .5 < threshold <= 1:
            raise ValueError('Invalid threshold')
        if not isinstance(metadata['model_version'], str) or not metadata['model_version']:
            raise ValueError('Missing model version')
        aspect = metadata.get('aspect_range', [.15, 6])
        if len(aspect) != 2 or not np.isfinite(aspect).all() or not 0 < aspect[0] <= aspect[1]:
            raise ValueError('Invalid aspect range')
        with np.load(directory / 'model.npz', allow_pickle=False) as arrays:
            if set(arrays.files) != {'coef', 'intercept', 'mean', 'scale'}:
                raise ValueError('Unexpected arrays')
            values = {name: arrays[name].copy() for name in arrays.files}
        n_features = features(np.zeros((64, 64, 3), np.uint8)).size
        for key, value in values.items():
            if value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise ValueError('Non-numeric weights')
            if value.shape != ((1,) if key == 'intercept' else (n_features,)):
                raise ValueError('Invalid weight dimensions')
        if np.any(values['scale'] <= 0):
            raise ValueError('Invalid scale')
        return SafeModel(metadata=metadata, **values)
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        raise ValueError('Invalid or missing safe model artifact') from exc


def predict_crop(model: SafeModel, image_bgr: np.ndarray) -> dict:
    if not valid_crop(image_bgr):
        return dict(state='unknown', confidence=0., open_probability=.5)
    aspect = image_bgr.shape[1] / image_bgr.shape[0]
    minimum, maximum = model.metadata.get('aspect_range', [.15, 6])
    if not minimum / 3 <= aspect <= maximum * 3:
        return dict(state='unknown', confidence=0., open_probability=.5)
    vector = (features(image_bgr) - model.mean) / model.scale
    logit = float(np.dot(vector, model.coef) + model.intercept[0])
    probability = float(1 / (1 + np.exp(-np.clip(logit, -700, 700))))
    confidence = max(probability, 1 - probability)
    state = ('open' if probability >= .5 else 'closed') if confidence >= model.metadata['threshold'] else 'unknown'
    return dict(state=state, confidence=confidence, open_probability=probability)
