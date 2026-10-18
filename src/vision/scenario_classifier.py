"""Experimental weak-label scenario candidates. No source-label lookup at inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .scenario_features import BACKBONE_SHA256, FEATURE_DIM, FEATURE_VERSION, SAMPLE_FPS, extract_video_features
from .video_io import iter_sampled_frames


@dataclass
class ScenarioModel:
    coef: np.ndarray
    intercept: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    supported: np.ndarray
    metadata: dict


def load_model(directory) -> ScenarioModel:
    directory = Path(directory)
    try:
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        artifact = (directory / 'model.npz').read_bytes()
        if metadata.get('artifact_sha256') != hashlib.sha256(artifact).hexdigest():
            raise ValueError('Head checksum mismatch')
        if (metadata.get('schema_version') != 1 or metadata.get('feature_version') != FEATURE_VERSION
                or metadata.get('backbone_sha256') != BACKBONE_SHA256):
            raise ValueError('Unsupported feature pipeline')
        classes = metadata['classes']
        if not isinstance(classes, list) or not 1 <= len(classes) <= 50:
            raise ValueError('Invalid scenario class list')
        types = set()
        for row in classes:
            for key in ('type', 'title'):
                if not isinstance(row.get(key), str) or not row[key].strip() or len(row[key]) > 500:
                    raise ValueError('Invalid class metadata')
            if row['type'] in types:
                raise ValueError('Duplicate class metadata')
            types.add(row['type'])
            for key in ('positive_groups', 'negative_groups'):
                if isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] < 0:
                    raise ValueError('Invalid training support')
        threshold = metadata['threshold']
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not .5 < threshold <= 1:
            raise ValueError('Invalid threshold')
        if not isinstance(metadata.get('model_version'), str) or not metadata['model_version']:
            raise ValueError('Missing model version')
        with np.load(directory / 'model.npz', allow_pickle=False) as saved:
            names = {'coef', 'intercept', 'mean', 'scale', 'supported'}
            if set(saved.files) != names:
                raise ValueError('Unexpected head arrays')
            arrays = {key: saved[key].copy() for key in names}
        count = len(classes)
        dimensions = dict(coef=(count, FEATURE_DIM), intercept=(count,), mean=(FEATURE_DIM,), scale=(FEATURE_DIM,), supported=(count,))
        for key, value in arrays.items():
            if value.shape != dimensions[key] or not np.isfinite(value).all():
                raise ValueError('Invalid head dimensions or values')
            if key == 'supported' and value.dtype.kind != 'b':
                raise ValueError('Invalid support flags')
            if key != 'supported' and value.dtype.kind != 'f':
                raise ValueError('Invalid numeric weights')
        if np.any(arrays['scale'] <= 0):
            raise ValueError('Invalid scale')
        expected = np.array([row['positive_groups'] > 0 and row['negative_groups'] > 0 for row in classes])
        if not np.array_equal(arrays['supported'], expected):
            raise ValueError('Class support and availability disagree')
        return ScenarioModel(metadata=metadata, **arrays)
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        raise ValueError('Invalid or missing scenario model') from exc


def score_features(model: ScenarioModel, timestamps, features) -> tuple[list[dict], np.ndarray]:
    times, vectors = np.asarray(timestamps, dtype=float), np.asarray(features, dtype=float)
    if (times.ndim != 1 or not len(times) or vectors.shape != (len(times), FEATURE_DIM)
            or not np.isfinite(times).all() or not np.isfinite(vectors).all()
            or (len(times) > 1 and np.any(np.diff(times) <= 0))):
        raise ValueError('Invalid video features')
    logits = ((vectors - model.mean) / model.scale) @ model.coef.T + model.intercept
    scores = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    scores[:, ~model.supported] = np.nan
    results = []
    threshold = float(model.metadata['threshold'])
    for index, definition in enumerate(model.metadata['classes']):
        peak = int(np.argmax(scores[:, index])) if model.supported[index] else None
        score = float(scores[peak, index]) if peak is not None else None
        results.append(dict(type=definition['type'], title=definition['title'], score=score, threshold=threshold,
                            decision='unsupported' if score is None else 'candidate' if score >= threshold else 'not_flagged',
                            positive_groups=definition['positive_groups'], negative_groups=definition['negative_groups'],
                            peak_sec=float(times[peak]) if peak is not None else None))
    return results, scores


def peak_interval(timestamps, scores, threshold) -> tuple[int, int, int]:
    peak = int(np.argmax(scores))
    start = end = peak
    while start > 0 and scores[start - 1] >= threshold and timestamps[start] - timestamps[start - 1] <= 1.5:
        start -= 1
    while end + 1 < len(scores) and scores[end + 1] >= threshold and timestamps[end + 1] - timestamps[end] <= 1.5:
        end += 1
    return start, peak, end


def analyze_video(video_path: str) -> dict:
    directory = Path(os.environ.get('MODEL_PATH') or os.environ.get('SCENARIO_MODEL_PATH') or '')
    model = load_model(directory)
    cv2.setNumThreads(4)
    timestamps, vectors = extract_video_features(video_path, directory / 'backbone.onnx')
    classes, scores = score_features(model, timestamps, vectors)
    events, needed = [], set()
    for index, result in enumerate(classes):
        if result['decision'] != 'candidate':
            continue
        start, peak, end = peak_interval(timestamps, scores[:, index], result['threshold'])
        points = sorted({float(timestamps[k]) for k in (start, peak, end)})
        needed.update(points)
        events.append(dict(event_type=result['type'], timecode_sec=float(timestamps[peak]),
                           start_sec=float(timestamps[start]), end_sec=float(timestamps[end]), confidence=result['score'],
                           description=f"Кандидат: {result['title']}. Показано окно с наибольшей оценкой сходства с обучающими примерами. Границы окна не являются подтверждёнными границами действия; причину нарушения проверяет человек.",
                           evidence_frames=[], _points=points))
    evidence = {}
    if needed:
        for timestamp, frame in iter_sampled_frames(video_path, sample_fps=SAMPLE_FPS):
            if timestamp not in needed:
                continue
            height, width = frame.shape[:2]
            factor = min(1., 960 / max(height, width))
            resized = cv2.resize(frame, (round(width * factor), round(height * factor)))
            ok, encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError('Evidence cannot be encoded')
            evidence[timestamp] = dict(timestamp_sec=float(timestamp), jpeg_bytes=encoded.tobytes())
        if set(evidence) != needed:
            raise ValueError('Required evidence frame is missing')
    for event in events:
        event['evidence_frames'] = [evidence[t] for t in event.pop('_points')]
    return dict(analysis_scope='bank_scenarios', verdict='review_required' if events else 'no_candidate',
                model_version=model.metadata['model_version'], feature_version=FEATURE_VERSION,
                score_semantics='uncalibrated_model_score',
                summary=dict(sample_count=len(timestamps), sample_fps=SAMPLE_FPS, events_truncated=False),
                class_results=classes, events=events, limitations=[
                    'Экспериментальная модель обучена на слабых метках таблицы; спорные причины не стали подтверждёнными.',
                    'Оценка отражает сходство с примерами и не является вероятностью доказанного нарушения.',
                    'На тип приходится мало независимых примеров; один пример не подтверждает перенос на новые видео.',
                    'Документы, содержание отчёта, подключение к сети, полномочия и содержимое сумок по этим признакам не проверяются.',
                    'Отсутствие кандидатов не означает отсутствие нарушений. Короткие действия между отсчётами 1 FPS могут быть пропущены.',
                    'Финальная модель обучена на всём наборе. Проверка её обучающих видео не является независимой оценкой качества.',
                ])
