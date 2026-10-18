"""Local numeric before/after audit. Does not export frames or source descriptions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.training.scenario_dataset import load_dataset
from src.training.train_scenarios import overall_binary_metrics
from src.vision.scenario_classifier import load_model, score_features, peak_interval
from src.vision.scenario_features import RESIDUAL_FEATURE_VERSION, scene_residual_features


def summarize(comparison):
    result = {}
    for scope in ('in_sample', 'out_of_fold'):
        rows = comparison['rows']
        unique = {r['duplicate_group_id']: r for r in rows}
        normals = [r for r in rows if not r['table_violation']]
        unique_normals = [r for r in unique.values() if not r['table_violation']]
        result[scope] = dict(
            normal_alarm_rows=sum(bool(r[scope]['candidate_types']) for r in normals),
            normal_rows=len(normals),
            normal_alarm_groups=sum(bool(r[scope]['candidate_types']) for r in unique_normals),
            normal_groups=len(unique_normals),
            micro=comparison[scope]['micro'],
            unique_group_micro=comparison[scope].get('unique_group', {}).get('micro'),
            overall=overall_binary_metrics(rows, scope))
    return result


def diagnose(manifest, cache, model_paths, output):
    dataset = load_dataset(manifest, cache)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, reports, comparisons = [], {}, {}
    diagnostic_ids = {'X-1', 'X-2', 'X-3', 'X-4', 'X-5', 'X-8'}
    for name, path in model_paths.items():
        model = load_model(path)
        comparison = json.loads((Path(path) / 'comparison.json').read_text(encoding='utf-8'))
        comparisons[name] = comparison
        by_row = {r['row_id']: r for r in comparison['rows']}
        difference, matching, broad, candidates, constant_normals = 0., 0, 0, 0, 0
        for group in dataset['groups']:
            features = group['features']
            if model.metadata['feature_version'] == RESIDUAL_FEATURE_VERSION:
                features = scene_residual_features(features)
            classes, scores = score_features(model, group['timestamps'], features)
            if not group['positive']:
                constant = np.median(features, axis=0)[None].copy()
                constant[:, 512:] = 0
                static, _ = score_features(model, np.array([0.]), constant)
                constant_normals += int(any(r['decision'] == 'candidate' for r in static))
            for row_id in group['row_ids']:
                row = by_row[row_id]
                frozen = {c['type']: c for c in row['in_sample']['class_results']}
                matching += int(all(c['decision'] == frozen[c['type']]['decision'] and
                                    c['peak_sec'] == frozen[c['type']]['peak_sec'] for c in classes))
                for index, c in enumerate(classes):
                    old = frozen[c['type']]
                    if c['score'] is not None:
                        difference = max(difference, abs(c['score'] - old['score']))
                    start = end = fraction = None
                    if c['decision'] == 'candidate':
                        first, _, last = peak_interval(group['timestamps'], scores[:, index], c['threshold'])
                        start, end = float(group['timestamps'][first]), float(group['timestamps'][last])
                        fraction = float(np.mean(scores[:, index] >= c['threshold']))
                        candidates += 1
                        broad += int(fraction >= .5)
                    records.append(dict(model=name, model_version=model.metadata['model_version'],
                        evaluation='known_training_replay', row_id=row_id, diagnostic=row_id in diagnostic_ids,
                        type=c['type'], expected=(c['type'] in row['expected_types']) if row['table_violation'] else False,
                        truth=('positive' if c['type'] in row['expected_types'] else 'unknown') if row['table_violation'] else 'negative',
                        decision=c['decision'], score=c['score'], peak_sec=c['peak_sec'], start_sec=start, end_sec=end,
                        threshold_fraction=fraction, positive_groups=c['positive_groups']))
        reports[name] = dict(model_version=model.metadata['model_version'], feature_version=model.metadata['feature_version'],
            metrics=summarize(comparison), replay_matching_decisions_and_peaks_rows=matching,
            replay_max_score_difference=difference, constant_feature_normal_alarm_groups=constant_normals,
            candidate_pairs_covering_at_least_half_samples=broad, candidate_pairs=candidates)
    baseline_name = next(iter(comparisons))
    baseline = {r['row_id']: r for r in comparisons[baseline_name]['rows']}
    for name, comparison in comparisons.items():
        common = {side: dict(tp=0, fn=0, fp=0, tn=0, positive=0, negative=0) for side in ('baseline', 'candidate')}
        for row in comparison['rows']:
            old = {c['type']: c for c in baseline[row['row_id']]['out_of_fold']['class_results']}
            for new in row['out_of_fold']['class_results']:
                prior = old[new['type']]
                if new['truth'] is None or new['decision'] == 'unsupported' or prior['decision'] == 'unsupported':
                    continue
                for side, item in [('baseline', prior), ('candidate', new)]:
                    positive, flagged = bool(new['truth']), item['decision'] == 'candidate'
                    common[side]['positive' if positive else 'negative'] += 1
                    common[side]['tp' if positive and flagged else 'fn' if positive else 'fp' if flagged else 'tn'] += 1
        reports[name]['common_supported_out_of_fold_pairs'] = common
    fields = list(records[0])
    with (output/'before-after.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    report = dict(evaluation_caveat='Exploratory grouped evaluation on known development corpus, not a fresh independent test; models were not selected for deployment.',
                  models=reports, diagnostics=[r for r in records if r['diagnostic']],
                  inference_inputs='pixels only; labels/row IDs used in this offline evaluation only')
    (output/'diagnostics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({name: {k:v for k,v in value.items() if k != 'metrics'} for name,value in reports.items()}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--model', action='append', required=True, help='NAME=MODEL_DIRECTORY; baseline first')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    diagnose(args.manifest, args.cache, dict(item.split('=',1) for item in args.model), args.output)
