"""Fixed-policy grouped evaluation and final weak scenario heads on cached pixels.

Run: python -m src.training.train_scenarios --manifest PATH --feature-cache DIR --output DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.training.scenario_dataset import FEATURE_DIM, load_dataset, label_mask

REGULARIZATION = .01
THRESHOLD = .65
ANCHOR_TOLERANCE_SEC = 5.
LIMITATIONS = [
    'Экспериментальные кандидаты по исходной таблице; спорные метки не становятся подтверждёнными событиями.',
    'Оценки модели не калиброваны и не являются вероятностью доказанного нарушения.',
    'Для большинства классов есть только одно исходное видео; при исключении этого видео класс не поддерживается.',
    'Проверка исключает целую группу идентичных видео; новая камера, освещение и независимые съёмки не проверены.',
    'Отсутствие кандидата не доказывает отсутствие нарушения или ненаблюдаемого действия.',
    'Точность времени измеряется относительно слабой отметки таблицы с допуском ±5 секунд, а не границ события.',
    'Повторный прогон всех обучающих видео является оценкой на обучающем наборе, отдельно от отложенной проверки.',
    'Строки с разными причинами на одном исходном видео сохраняются в сравнении; обучение использует объединение причин.',
]


def class_samples(dataset, type_id, excluded_groups=()):
    """Equal mass per video inside each class, equal positive/negative mass."""
    excluded = set(excluded_groups)
    selected = []
    for group in dataset['groups']:
        if group['group_id'] in excluded:
            continue
        labels = label_mask(group, type_id)
        mask = labels >= 0
        if mask.any():
            selected.append((group, labels[mask], group['features'][mask]))
    positives = sum(int(labels[0] == 1) for _, labels, _ in selected)
    negatives = len(selected) - positives
    vectors, labels_list, weights, ids = [], [], [], []
    for group, labels, values in selected:
        count = positives if labels[0] == 1 else negatives
        # Total effective sample mass is number of independently labelled groups.
        weight = len(selected) / (2 * count * len(labels))
        vectors.append(values)
        labels_list.append(labels)
        weights.append(np.full(len(labels), weight, dtype=np.float64))
        ids.extend([group['group_id']] * len(labels))
    return dict(features=np.concatenate(vectors) if vectors else np.empty((0, FEATURE_DIM)),
                labels=np.concatenate(labels_list) if labels_list else np.empty(0, dtype=np.int8),
                weights=np.concatenate(weights) if weights else np.empty(0),
                group_ids=np.asarray(ids), positive_groups=positives, negative_groups=negatives)


def fit_heads(dataset, excluded_groups=()):
    """Fit scaling and every head using only permitted original groups."""
    excluded = set(excluded_groups)
    train_groups = [g for g in dataset['groups'] if g['group_id'] not in excluded]
    # Equal group weighting also keeps long clips from dominating scaling.
    first_moments, second_moments = [], []
    for group in train_groups:
        eligible = np.zeros(len(group['timestamps']), dtype=bool)
        for definition in dataset['classes']:
            eligible |= label_mask(group, definition['type']) >= 0
        values = group['features'][eligible]
        if len(values):
            first_moments.append(values.mean(axis=0))
            second_moments.append(np.mean(values ** 2, axis=0))
    mean = np.mean(first_moments, axis=0) if first_moments else np.zeros(FEATURE_DIM)
    variance = np.maximum(np.mean(second_moments, axis=0) - mean ** 2, 0) if second_moments else np.ones(FEATURE_DIM)
    scale = np.sqrt(variance)
    scale = np.where(scale < 1e-6, 1., scale)
    count = len(dataset['classes'])
    coef, intercept, supported = np.zeros((count, FEATURE_DIM)), np.zeros(count), np.zeros(count, dtype=bool)
    definitions = []
    for index, definition in enumerate(dataset['classes']):
        samples = class_samples(dataset, definition['type'], excluded)
        support = dict(positive_groups=samples['positive_groups'], negative_groups=samples['negative_groups'])
        definitions.append(dict(definition, **support))
        if min(support.values()) == 0:
            continue
        model = LogisticRegression(C=REGULARIZATION, solver='liblinear', random_state=0,
                                   max_iter=2000, tol=1e-8)
        model.fit((samples['features'] - mean) / scale, samples['labels'], sample_weight=samples['weights'])
        coef[index], intercept[index] = model.coef_[0], model.intercept_[0]
        supported[index] = True
    return dict(coef=coef, intercept=intercept, mean=mean, scale=scale, supported=supported,
                classes=definitions, threshold=THRESHOLD,
                training_groups=[g['group_id'] for g in train_groups])


def score_group(model, group):
    """Max-window decision shared with inference: equality is a candidate."""
    vectors = (group['features'] - model['mean']) / model['scale']
    logits = vectors @ model['coef'].T + model['intercept']
    scores = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    results = []
    for index, definition in enumerate(model['classes']):
        supported = bool(model['supported'][index])
        peak = int(np.argmax(scores[:, index])) if supported else None
        score = float(scores[peak, index]) if supported else None
        decision = ('candidate' if score >= THRESHOLD else 'not_flagged') if supported else 'unsupported'
        results.append(dict(definition, supported=supported, score=score,
                            peak_sec=float(group['timestamps'][peak]) if supported else None,
                            threshold=THRESHOLD, decision=decision))
    return results


def binary_metrics(entries):
    """Only positively listed types and explicit normal videos have known truth."""
    known = [e for e in entries if e['truth'] is not None]
    scored = [e for e in known if e['decision'] != 'unsupported']
    positive = [e for e in scored if e['truth']]
    negative = [e for e in scored if not e['truth']]
    tp = sum(e['decision'] == 'candidate' for e in positive)
    fp = sum(e['decision'] == 'candidate' for e in negative)
    fn, tn = len(positive) - tp, len(negative) - fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(positive) if positive else None
    anchor_entries = [e for e in positive if e.get('anchor_hit') is not None]
    return dict(known_truth=len(known), unknown_truth=len(entries) - len(known),
                positive_truth=sum(bool(e['truth']) for e in known),
                negative_truth=sum(not e['truth'] for e in known),
                unsupported=len(known) - len(scored), scored=len(scored),
                scored_positive=len(positive), scored_negative=len(negative),
                tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                accuracy=(tp + tn) / len(scored) if scored else None,
                coverage=len(scored) / len(known) if known else None,
                anchor_hits=sum(bool(e['anchor_hit']) for e in anchor_entries),
                anchor_evaluated=len(anchor_entries),
                anchor_hit_rate=sum(bool(e['anchor_hit']) for e in anchor_entries) / len(anchor_entries) if anchor_entries else None)


def overall_binary_metrics(rows, scope):
    """Any-candidate screening, with unsupported expected causes excluded.

This deliberately does not measure cause correctness; per-type metrics do.
Normal rows need all heads supported, since any head could produce a candidate.
"""
    entries = []
    for row in rows:
        results = row[scope]['class_results']
        by_type = {entry['type']: entry for entry in results}
        needed = row['expected_types'] if row['table_violation'] else list(by_type)
        supported = bool(needed) and all(t in by_type and by_type[t]['decision'] != 'unsupported' for t in needed)
        decision = ('candidate' if any(e['decision'] == 'candidate' for e in results) else 'not_flagged') if supported else 'unsupported'
        entries.append(dict(truth=row['table_violation'], decision=decision))
    metrics = binary_metrics(entries)
    metrics['interpretation'] = 'Any candidate versus table-positive clip; cause correctness is evaluated separately.'
    metrics['eligibility'] = 'Positive rows: every expected type supported; normal rows: every model class supported.'
    return metrics


def _annotate_scores(results, expected, normal, anchors, evaluation):
    entries = []
    for result in results:
        truth = True if result['type'] in expected else False if normal else None
        matching = [a['seconds'] for a in anchors if result['type'] in a['type_ids']]
        distance = min(abs(result['peak_sec'] - time) for time in matching) if matching and result['peak_sec'] is not None else None
        hit = result['decision'] == 'candidate' and distance <= ANCHOR_TOLERANCE_SEC if distance is not None else None
        entries.append(dict(result, truth=truth, anchor_hit=hit, anchor_distance_sec=distance))
    return dict(evaluation=evaluation, class_results=entries,
                candidate_types=[e['type'] for e in entries if e['decision'] == 'candidate'],
                unsupported_types=[e['type'] for e in entries if e['decision'] == 'unsupported'])


def _aggregate(entries, classes):
    return dict(per_type={c['type']: binary_metrics([e for e in entries if e['type'] == c['type']]) for c in classes},
                micro=binary_metrics(entries))


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def train(manifest_path, feature_cache, output_dir):
    dataset = load_dataset(manifest_path, feature_cache)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    heldout = {}
    for fold_index, group in enumerate(dataset['groups'], start=1):
        fold = fit_heads(dataset, {group['group_id']})
        heldout[group['group_id']] = score_group(fold, group)
        print(json.dumps(dict(fold=fold_index, total=len(dataset['groups']), heldout=group['row_ids'])), flush=True)
    # Persist frozen held-out decisions before fitting a separate deployment head.
    _write_json(output / 'out-of-fold-predictions.json', dict(
        evaluation='leave_one_media_group_out', feature_version=dataset['feature_version'],
        manifest_sha256=dataset['manifest_sha256'], selected_C=REGULARIZATION,
        threshold=THRESHOLD, groups=heldout))
    model = fit_heads(dataset)
    in_sample = {g['group_id']: score_group(model, g) for g in dataset['groups']}
    version_hash = hashlib.sha256()
    for name in ('coef', 'intercept', 'mean', 'scale'):
        version_hash.update(model[name].astype('<f8').tobytes())
    version_hash.update(model['supported'].tobytes())
    version_hash.update(json.dumps(dict(feature_version=dataset['feature_version'],
                                       backbone_sha256=dataset['backbone_sha256'], classes=model['classes'],
                                       threshold=THRESHOLD), sort_keys=True).encode())
    version = 'scenario-cpu-v1-' + version_hash.hexdigest()[:16]
    np.savez(output / 'model.npz', **{k: model[k] for k in ('coef', 'intercept', 'mean', 'scale', 'supported')})
    metadata = dict(schema_version=1, feature_version=dataset['feature_version'], model_version=version,
                    threshold=THRESHOLD, selected_C=REGULARIZATION, classes=model['classes'],
                    artifact_sha256=hashlib.sha256((output / 'model.npz').read_bytes()).hexdigest(),
                    backbone_sha256=dataset['backbone_sha256'], training_scope='all_available_weak_labels',
                    score_semantics='uncalibrated_model_score', limitations=LIMITATIONS)
    _write_json(output / 'metadata.json', metadata)
    groups = {g['group_id']: g for g in dataset['groups']}
    rows = []
    for source in dataset['rows']:
        gid = source['group_id']
        anchors = [dict(seconds=source['seconds'], type_ids=source['table_type_ids'])] if source['seconds'] is not None else []
        row = {k: source.get(k) for k in ('row_id', 'video_id', 'filename', 'seconds', 'user_description',
                                         'table_violation', 'annotation_status', 'source_annotation_events')}
        row.update(expected_types=source['table_type_ids'], duplicate_group_id=gid,
                   duplicate_conflict=groups[gid]['table_type_sets_conflict'],
                   out_of_fold=_annotate_scores(heldout[gid], source['table_type_ids'], not source['table_violation'],
                                                anchors, 'leave_one_media_group_out'),
                   in_sample=_annotate_scores(in_sample[gid], source['table_type_ids'], not source['table_violation'],
                                               anchors, 'resubstitution'))
        rows.append(row)
    summary = dict(rows=len(rows), groups=len(groups), types=len(model['classes']),
                   positive_rows=sum(r['table_violation'] for r in rows),
                   negative_rows=sum(not r['table_violation'] for r in rows),
                   duplicate_conflict_rows=sum(r['duplicate_conflict'] for r in rows),
                   positive_groups=sum(g['positive'] for g in groups.values()),
                   negative_groups=sum(not g['positive'] for g in groups.values()))
    report = dict(schema_version=1, model_version=version, feature_version=dataset['feature_version'],
                  threshold=THRESHOLD, selected_C=REGULARIZATION, classes=model['classes'],
                  score_semantics='uncalibrated_model_score', summary=summary, limitations=LIMITATIONS,
                  manifest_sha256=dataset['manifest_sha256'], cache_metadata_sha256=dataset['cache_metadata_sha256'],
                  evaluation_policy=dict(method='leave_one_media_group_out', folds=len(groups),
                      class_truth='listed type positive; explicit normal clip negative; remaining labels unknown',
                      weights='equal group mass within each class and equal mass between classes; total weight equals labelled groups',
                      scaling='training groups only, equal group moments on labelled windows',
                      positive_anchor_radius_sec=3., anchor_tolerance_sec=ANCHOR_TOLERANCE_SEC,
                      hyperparameter_tuning=False, metric_unit='source_table_row; duplicate-safe unique_group metrics also provided'))
    for scope, predictions in [('out_of_fold', heldout), ('in_sample', in_sample)]:
        entries = [entry for row in rows for entry in row[scope]['class_results']]
        stats = _aggregate(entries, model['classes'])
        stats['overall_binary'] = overall_binary_metrics(rows, scope)
        stats['summary'] = dict(summary, evaluation='leave_one_media_group_out' if scope == 'out_of_fold' else 'resubstitution',
                                candidate_rows=sum(bool(row[scope]['candidate_types']) for row in rows),
                                rows_with_unsupported=sum(bool(row[scope]['unsupported_types']) for row in rows))
        unique_entries = [entry for group in groups.values() for entry in _annotate_scores(
            predictions[group['group_id']], group['training_type_ids'], not group['positive'],
            group['anchors'], scope)['class_results']]
        stats['unique_group'] = _aggregate(unique_entries, model['classes'])
        report[scope] = stats
    comparison = dict(schema_version=1, model_version=version, feature_version=dataset['feature_version'],
                      threshold=THRESHOLD, score_semantics='uncalibrated_model_score', classes=model['classes'],
                      summary=summary, limitations=LIMITATIONS, rows=rows,
                      out_of_fold=report['out_of_fold'], in_sample=report['in_sample'])
    _write_json(output / 'report.json', report)
    _write_json(output / 'comparison.json', comparison)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--feature-cache', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = train(args.manifest, args.feature_cache, args.output)
    print(json.dumps(dict(model_version=report['model_version'], summary=report['summary'],
                         out_of_fold=report['out_of_fold']['micro'], in_sample=report['in_sample']['micro']), indent=2))


if __name__ == '__main__':
    main()
