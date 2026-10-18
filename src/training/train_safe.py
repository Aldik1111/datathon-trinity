"""Train a deterministic CPU baseline without consulting held-out test labels.

Run: python -m src.training.train_safe --manifest PATH --output DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.vision.safe_model import FEATURE_VERSION, SafeModel, features

LABELS = ['closed', 'open']


def _validate_manifest(manifest):
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('samples'), list):
        raise ValueError('Unsupported dataset manifest')
    split_samples = {key: [] for key in ('train', 'val', 'test')}
    assignments, video_assignments = {}, {}
    for sample in manifest['samples']:
        split, label = sample.get('split'), sample.get('label')
        if split not in split_samples or label not in LABELS:
            raise ValueError('Invalid sample label or split')
        for field, registry in [('group_id', assignments), ('video_id', video_assignments)]:
            key = sample.get(field)
            if not isinstance(key, str) or not key:
                raise ValueError('Missing video/group identity for split isolation')
            if key in registry and registry[key] != split:
                raise ValueError('Dataset group/video leaks across splits')
            registry[key] = split
        split_samples[split].append(sample)
    if any(not samples for samples in split_samples.values()):
        raise ValueError('Train, validation and test splits must all contain samples')
    if {sample['label'] for sample in split_samples['train']} != set(LABELS):
        raise ValueError('Training requires both open and closed samples')
    return split_samples


def _read_samples(samples):
    vectors, labels, aspects = [], [], []
    for sample in samples:
        image = cv2.imread(str(sample['image_path']), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Dataset image cannot be decoded')
        vectors.append(features(image))
        labels.append(int(sample['label'] == 'open'))
        aspects.append(image.shape[1] / image.shape[0])
    return np.stack(vectors), np.asarray(labels), aspects


def _scores(vectors, coef, intercept):
    logits = vectors @ coef + intercept[0]
    return 1 / (1 + np.exp(-np.clip(logits, -700, 700)))


def _metrics(labels, probabilities, threshold, samples):
    prediction = (probabilities >= .5).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(labels, prediction, labels=[0, 1], zero_division=0)
    covered = np.maximum(probabilities, 1 - probabilities) >= threshold
    return dict(samples=len(samples), videos=len({s['video_id'] for s in samples}),
                groups=len({s['group_id'] for s in samples}),
                class_counts={name: int(np.sum(labels == index)) for index, name in enumerate(LABELS)},
                confusion_matrix=confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
                confusion_matrix_labels=LABELS,
                per_class={name: dict(precision=float(precision[index]), recall=float(recall[index]),
                                      f1=float(f1[index]), support=int(support[index]))
                           for index, name in enumerate(LABELS)},
                accuracy=float(accuracy_score(labels, prediction)), macro_f1=float(np.mean(f1)),
                coverage=float(covered.mean()), abstained=int((~covered).sum()),
                covered_accuracy=float(np.mean(prediction[covered] == labels[covered])) if covered.any() else None)


def train(manifest_path, output_dir):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    splits = _validate_manifest(manifest)
    train_x, train_y, aspects = _read_samples(splits['train'])
    val_x, val_y, _ = _read_samples(splits['val'])
    # Fit preprocessing on training examples only.
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-6, 1., scale)
    train_x, val_x = (train_x - mean) / scale, (val_x - mean) / scale
    grid, best, best_f1 = [], None, -1.
    for regularization in (.001, .01, .1):
        estimator = LogisticRegression(C=regularization, solver='liblinear', class_weight='balanced',
                                       random_state=0, max_iter=2000, tol=1e-8)
        estimator.fit(train_x, train_y)
        probability = estimator.predict_proba(val_x)[:, 1]
        metrics = _metrics(val_y, probability, .8, splits['val'])
        grid.append(dict(C=regularization, validation_macro_f1=metrics['macro_f1']))
        # Ties favor the stronger regularization visited first.
        if metrics['macro_f1'] > best_f1 + 1e-12:
            best, best_f1 = estimator, metrics['macro_f1']
    val_probability = best.predict_proba(val_x)[:, 1]
    thresholds = []
    for threshold in (.6, .7, .8, .9):
        metrics = _metrics(val_y, val_probability, threshold, splits['val'])
        thresholds.append(dict(threshold=threshold, coverage=metrics['coverage'],
                               covered_accuracy=metrics['covered_accuracy'],
                               covered_samples=metrics['samples'] - metrics['abstained']))
    acceptable = [row for row in thresholds if row['covered_samples'] >= 3 and row['covered_accuracy'] >= .95]
    threshold = max(acceptable, key=lambda row: (row['coverage'], -row['threshold']))['threshold'] if acceptable else .9
    coef, intercept = best.coef_[0].astype(np.float64), best.intercept_.astype(np.float64)
    version_digest = hashlib.sha256()
    for array in (coef, intercept, mean, scale):
        version_digest.update(array.astype('<f8').tobytes())
    version_digest.update(f'{FEATURE_VERSION}:{threshold}'.encode())
    version = 'safe-cpu-v1-' + version_digest.hexdigest()[:16]
    metadata = dict(schema_version=1, feature_version=FEATURE_VERSION, model_version=version,
                    threshold=threshold, selected_C=float(best.C), class_labels=LABELS,
                    score_semantics='uncalibrated_model_score', aspect_range=[min(aspects), max(aspects)],
                    numpy_version=np.__version__, opencv_version=cv2.__version__, sklearn_version=sklearn.__version__)
    # The fixed test partition is first loaded/evaluated after weights and
    # threshold are frozen. Its metrics never feed back into model selection.
    test_x, test_y, _ = _read_samples(splits['test'])
    test_x = (test_x - mean) / scale
    report = dict(model_version=version, score_semantics='uncalibrated_model_score',
                  threshold=threshold, selected_C=float(best.C), source_sha256=manifest.get('source_sha256'),
                  selection=dict(regularization_grid=grid, validation_threshold_grid=thresholds,
                                 rule='max validation macro F1; stronger regularization on ties; maximum coverage at >=95% observed covered accuracy with >=3 validation samples; otherwise threshold 0.9',
                                 minimum_validation_accuracy_met=bool(acceptable), fixed_test_evaluations=1),
                  splits={}, limitations=[
                      'Small, sparse, same-camera labels do not establish reliable generalization to new cameras or lighting.',
                      'Open/closed only within an explicitly selected static ROI; unknown is confidence abstention, not a learned class.',
                      'Logistic outputs are uncalibrated model scores; confidence can be high on unfamiliar images.',
                      'Frame classification metrics do not measure event interval accuracy; sparse labels provide no event ground truth.',
                      'No identity, people, transfers, lock operation, or regulatory violation recognition.',
                      'Validation threshold accuracy is an observed selection statistic, not a performance guarantee.',
                  ])
    for split, vectors, labels in [('train', train_x, train_y), ('val', val_x, val_y), ('test', test_x, test_y)]:
        report['splits'][split] = _metrics(labels, _scores(vectors, coef, intercept), threshold, splits[split])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'model.npz', coef=coef, intercept=intercept, mean=mean, scale=scale)
    metadata['artifact_sha256'] = hashlib.sha256((output / 'model.npz').read_bytes()).hexdigest()
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
