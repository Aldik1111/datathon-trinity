"""Protect weak labels, original-group isolation, and honest denominators."""
import hashlib
import importlib
import importlib.util
import json

import numpy as np
import pytest


def modules():
    for name in ('src.training.scenario_dataset', 'src.training.train_scenarios'):
        assert importlib.util.find_spec(name) is not None, f'Missing implementation: {name}'
    return (importlib.import_module('src.training.scenario_dataset'),
            importlib.import_module('src.training.train_scenarios'))


def fixture(tmp_path):
    cache = tmp_path / 'features'
    cache.mkdir()
    (cache / 'metadata.json').write_text(json.dumps({
        'feature_version': 'mobilenetv2-gap-temporal-v1', 'backbone_sha256': 'a' * 64}))
    classes = ['door_left_open', 'safe_key_left']
    rows, groups = [], []
    specifications = [('positive', [classes[0]], 3, 8.),
                      ('duplicate', [classes[1]], 9, 8.),
                      ('positive2', [classes[0]], 3, 6.),
                      ('negative', [], None, -4.),
                      ('negative2', [], None, -6.)]
    by_group = {}
    for i, (name, types, seconds, value) in enumerate(specifications):
        group = hashlib.sha256(('positive' if name == 'duplicate' else name).encode()).hexdigest()
        row = dict(row_id=name, number=i, video_id=name, filename=name + '.mov',
                   seconds=seconds, user_description=name, table_violation=bool(types),
                   table_type_ids=types, annotation_status='review' if types else 'done',
                   source_annotation_events=[], group_id=group,
                   original_media=dict(path=str(tmp_path / (name + '.mov')), available_local=True,
                                       sha256=group, hash_verified_local=True))
        rows.append(row)
        by_group.setdefault(group, []).append(row)
        if name != 'duplicate':
            times = np.arange(13, dtype=float)
            features = np.zeros((13, 768), dtype=np.float32)
            features[:, :4] = value
            features[:, 4] = times / 10
            np.savez(cache / (group + '.npz'), timestamps=times, features=features)
    for group, items in by_group.items():
        groups.append(dict(group_id=group, row_ids=[r['row_id'] for r in items],
                           canonical_row_id=items[0]['row_id'],
                           local_original_path=items[0]['original_media']['path'],
                           positive=any(r['table_violation'] for r in items),
                           training_type_ids=sorted({t for r in items for t in r['table_type_ids']}),
                           table_type_sets_conflict=len({tuple(r['table_type_ids']) for r in items}) > 1,
                           anchors=[dict(row_id=r['row_id'], seconds=r['seconds'],
                                         type_ids=r['table_type_ids']) for r in items if r['seconds'] is not None]))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, target_type_ids=classes, rows=rows, groups=groups)))
    return manifest, cache


def test_anchor_union_keeps_other_positive_labels_unknown(tmp_path):
    data_api, _ = modules()
    path, cache = fixture(tmp_path)
    data = data_api.load_dataset(path, cache)
    group = next(g for g in data['groups'] if len(g['row_ids']) == 2)
    np.testing.assert_array_equal(data_api.label_mask(group, 'door_left_open'),
                                  [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(data_api.label_mask(group, 'safe_key_left'),
                                  [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1])
    unknown = next(g for g in data['groups'] if g['row_ids'] == ['positive2'])
    assert (data_api.label_mask(unknown, 'safe_key_left') == -1).all()
    normal = next(g for g in data['groups'] if g['row_ids'] == ['negative'])
    assert (data_api.label_mask(normal, 'safe_key_left') == 0).all()


def test_class_weights_balance_groups_and_classes_not_clip_length(tmp_path):
    data_api, training = modules()
    path, cache = fixture(tmp_path)
    data = data_api.load_dataset(path, cache)
    normal = next(g for g in data['groups'] if g['row_ids'] == ['negative'])
    normal['features'] = np.repeat(normal['features'], 10, axis=0)
    normal['timestamps'] = np.repeat(normal['timestamps'], 10)
    samples = training.class_samples(data, 'door_left_open')
    weights, labels, ids = samples['weights'], samples['labels'], samples['group_ids']
    assert weights[labels == 0].sum() == pytest.approx(weights[labels == 1].sum())
    negative_groups = sorted(set(ids[labels == 0]))
    assert weights[ids == negative_groups[0]].sum() == pytest.approx(weights[ids == negative_groups[1]].sum())
    assert samples['positive_groups'] == 2 and samples['negative_groups'] == 2


def test_excluded_group_changes_neither_scaling_nor_heads_and_singleton_abstains(tmp_path):
    data_api, training = modules()
    path, cache = fixture(tmp_path)
    data = data_api.load_dataset(path, cache)
    heldout = next(g for g in data['groups'] if len(g['row_ids']) == 2)
    first = training.fit_heads(data, excluded_groups={heldout['group_id']})
    heldout['features'][:] = 999999
    second = training.fit_heads(data, excluded_groups={heldout['group_id']})
    for name in ('coef', 'intercept', 'mean', 'scale', 'supported'):
        np.testing.assert_array_equal(first[name], second[name])
    assert first['supported'].tolist() == [True, False]
    assert not first['coef'][1].any() and first['intercept'][1] == 0
    results = training.score_group(first, heldout)
    assert results[1]['score'] is None and results[1]['peak_sec'] is None
    assert results[1]['decision'] == 'unsupported'


def test_comparison_preserves_duplicates_and_unsupported_denominators(tmp_path):
    _, training = modules()
    path, cache = fixture(tmp_path)
    before = path.read_bytes()
    output = tmp_path / 'model'
    report = training.train(path, cache, output)
    comparison = json.loads((output / 'comparison.json').read_text('utf-8'))
    assert path.read_bytes() == before
    assert report['summary']['rows'] == 5 and report['summary']['groups'] == 4
    assert report['summary']['duplicate_conflict_rows'] == 2
    assert len(comparison['rows']) == 5
    duplicated = [r for r in comparison['rows'] if r['duplicate_conflict']]
    assert duplicated[0]['out_of_fold']['candidate_types'] == duplicated[1]['out_of_fold']['candidate_types']
    assert all(r['out_of_fold']['evaluation'] == 'leave_one_media_group_out' for r in comparison['rows'])
    assert all(r['in_sample']['evaluation'] == 'resubstitution' for r in comparison['rows'])
    single = report['out_of_fold']['per_type']['safe_key_left']
    assert single['known_truth'] == 3 and single['unknown_truth'] == 2
    assert single['unsupported'] == 1 and single['scored'] == 2
    assert single['scored_positive'] == 0 and single['recall'] is None
    assert report['in_sample']['per_type']['safe_key_left']['unsupported'] == 0
    with np.load(output / 'model.npz', allow_pickle=False) as artifact:
        assert artifact['coef'].shape == (2, 768)
        assert artifact['supported'].dtype.kind == 'b'
        assert all(artifact[name].dtype.kind == 'f' for name in ('coef', 'intercept', 'mean', 'scale'))
    metadata = json.loads((output / 'metadata.json').read_text('utf-8'))
    assert metadata['artifact_sha256'] == hashlib.sha256((output / 'model.npz').read_bytes()).hexdigest()
    assert metadata['threshold'] == .65 and metadata['training_scope'] == 'all_available_weak_labels'


def test_known_truth_metrics_do_not_turn_abstention_or_unknown_into_negative():
    _, training = modules()
    entries = [dict(type='door_left_open', truth=True, decision='candidate', anchor_hit=True),
               dict(type='door_left_open', truth=True, decision='unsupported', anchor_hit=None),
               dict(type='door_left_open', truth=False, decision='candidate', anchor_hit=None),
               dict(type='door_left_open', truth=False, decision='not_flagged', anchor_hit=None),
               dict(type='door_left_open', truth=None, decision='candidate', anchor_hit=None)]
    metrics = training.binary_metrics(entries)
    assert {k: metrics[k] for k in ('tp', 'fp', 'tn', 'fn', 'known_truth', 'unknown_truth', 'unsupported', 'scored')} == {
        'tp': 1, 'fp': 1, 'tn': 1, 'fn': 0, 'known_truth': 4, 'unknown_truth': 1, 'unsupported': 1, 'scored': 3}
    assert metrics['precision'] == .5 and metrics['recall'] == 1.
    assert metrics['coverage'] == .75
    assert metrics['anchor_hits'] == 1 and metrics['anchor_evaluated'] == 1


def test_overall_binary_requires_support_for_every_expected_type_or_normal_class():
    _, training = modules()
    assert hasattr(training, 'overall_binary_metrics'), 'Missing coverage-aware binary evaluation'
    def row(positive, expected, decisions):
        return dict(table_violation=positive, expected_types=expected,
                    out_of_fold=dict(class_results=[dict(type=t, decision=d) for t, d in decisions.items()]))
    rows = [row(True, ['a'], {'a': 'unsupported', 'b': 'candidate'}),
            row(True, ['a'], {'a': 'not_flagged', 'b': 'not_flagged'}),
            row(False, [], {'a': 'candidate', 'b': 'not_flagged'}),
            row(False, [], {'a': 'candidate', 'b': 'unsupported'}),
            row(True, ['a'], {'a': 'not_flagged', 'b': 'candidate'})]
    metrics = training.overall_binary_metrics(rows, 'out_of_fold')
    assert {k: metrics[k] for k in ('tp', 'fp', 'tn', 'fn', 'scored', 'unsupported')} == {
        'tp': 1, 'fp': 1, 'tn': 0, 'fn': 1, 'scored': 3, 'unsupported': 2}
    assert metrics['coverage'] == .6 and metrics['recall'] == .5


@pytest.mark.parametrize('fault', ['nan', 'version', 'group_membership', 'missing_anchor'])
def test_invalid_cache_or_group_provenance_fails_before_model_write(tmp_path, fault):
    data_api, training = modules()
    path, cache = fixture(tmp_path)
    manifest = json.loads(path.read_text())
    if fault == 'nan':
        cache_file = cache / (manifest['groups'][0]['group_id'] + '.npz')
        np.savez(cache_file, timestamps=np.arange(2.), features=np.full((2, 768), np.nan))
    elif fault == 'version':
        (cache / 'metadata.json').write_text(json.dumps(dict(feature_version='different', backbone_sha256='a' * 64)))
    elif fault == 'group_membership':
        manifest['groups'][0]['row_ids'] = ['negative']
        path.write_text(json.dumps(manifest))
    else:
        manifest['rows'][0]['seconds'] = None
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        training.train(path, cache, tmp_path / 'model')
    assert not (tmp_path / 'model' / 'model.npz').exists()
