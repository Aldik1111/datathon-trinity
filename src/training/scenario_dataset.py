"""Read cached image features and preserve weak, partial scenario labels.

Source media/annotations are never edited. A missing type in a positive clip is
unknown. Only the user's explicitly normal clips provide negative supervision.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURE_VERSION = 'mobilenetv2-gap-temporal-v1'
FEATURE_DIM = 768
POSITIVE_RADIUS_SEC = 3.0
TITLES = {
    'money_returned': 'Деньги возвращены клиенту',
    'identity_check_missing': 'Визуальная проверка документа не наблюдалась',
    'money_not_secured': 'Деньги остались вне хранения',
    'money_not_secured_before_client_left': 'Клиент ушёл, деньги остались без хранения',
    'unattended_screen': 'Светящийся экран без сотрудника',
    'cash_count_missing': 'Пересчёт денег не наблюдался',
    'report_record_missing': 'Запись в отчёте не наблюдалась',
    'cash_bag_unattended': 'Сумка с видимыми деньгами оставлена без сотрудника',
    'parallel_service': 'Параллельное обслуживание клиентов',
    'safe_left_open': 'Сейф оставлен открытым',
    'safe_key_left': 'Ключ оставлен в сейфе',
    'protected_items_unsecured': 'Защищаемые предметы оставлены открыто',
    'door_left_open': 'Дверь оставлена открытой',
    'entry_through_open_door': 'Вход через открытую дверь',
    'client_left_before_count_complete': 'Клиент ушёл до видимого завершения пересчёта',
    'personal_items_before_shift': 'Личные вещи остались перед сменой',
    'mixed_denominations': 'Смешение номиналов / валют / видов банкнот',
    'mixed_client_cash': 'Объединены деньги разных клиентских сцен',
}


def label_mask(group, type_id):
    """1 inside matching weak anchors, 0 for normal clips, -1 otherwise."""
    times = group['timestamps']
    labels = np.full(len(times), -1 if group['positive'] else 0, dtype=np.int8)
    for anchor in group['anchors']:
        if type_id in anchor['type_ids']:
            labels[np.abs(times - anchor['seconds']) <= POSITIVE_RADIUS_SEC] = 1
    return labels


def load_dataset(manifest_path, feature_cache):
    path, cache = Path(manifest_path), Path(feature_cache)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    metadata = json.loads((cache / 'metadata.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or metadata.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported manifest or feature version')
    backbone_sha = metadata.get('backbone_sha256', '')
    if not isinstance(backbone_sha, str) or not re.fullmatch('[a-f0-9]{64}', backbone_sha):
        raise ValueError('Invalid backbone checksum provenance')
    classes = manifest.get('target_type_ids', [])
    if not classes or len(set(classes)) != len(classes) or any(t not in TITLES for t in classes):
        raise ValueError('Invalid scenario class taxonomy')
    rows = manifest.get('rows', [])
    row_ids, by_group = set(), defaultdict(list)
    for row in rows:
        row_id, gid = row.get('row_id'), row.get('group_id')
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise ValueError('Invalid or duplicate source row identity')
        if not isinstance(gid, str) or not re.fullmatch('[a-f0-9]{64}', gid):
            raise ValueError('Invalid media group identity')
        types = row.get('table_type_ids')
        if (not isinstance(types, list) or len(set(types)) != len(types)
                or any(t not in classes for t in types)
                or type(row.get('table_violation')) is not bool
                or row['table_violation'] != bool(types)):
            raise ValueError('Invalid weak table labels')
        anchor = row.get('seconds')
        if types and (isinstance(anchor, bool) or not isinstance(anchor, (float, int))
                      or not math.isfinite(anchor) or anchor < 0):
            raise ValueError('Positive row needs a finite source anchor')
        original = row.get('original_media', {})
        if original.get('sha256') != gid or not original.get('available_local'):
            raise ValueError('Original media provenance is missing or inconsistent')
        row_ids.add(row_id)
        by_group[gid].append(row)
    declared = manifest.get('groups', [])
    if not rows or len(declared) != len(by_group):
        raise ValueError('Missing or duplicate media groups')
    seen, groups = set(), []
    for declaration in declared:
        gid = declaration.get('group_id')
        if gid in seen or gid not in by_group:
            raise ValueError('Invalid duplicate group identity')
        seen.add(gid)
        source_rows = by_group[gid]
        source_ids = {r['row_id'] for r in source_rows}
        if (set(declaration.get('row_ids', [])) != source_ids
                or len(declaration.get('row_ids', [])) != len(source_ids)):
            raise ValueError('Group membership disagrees with source rows')
        types = sorted({t for r in source_rows for t in r['table_type_ids']})
        if sorted(declaration.get('training_type_ids', [])) != types:
            raise ValueError('Group labels are not the union of source rows')
        if len({r['table_violation'] for r in source_rows}) != 1:
            raise ValueError('Same media has contradictory positive and normal labels')
        anchors = [dict(row_id=r['row_id'], seconds=float(r['seconds']), type_ids=r['table_type_ids'])
                   for r in source_rows if r['table_type_ids']]
        try:
            with np.load(cache / f'{gid}.npz', allow_pickle=False) as arrays:
                times = np.asarray(arrays['timestamps'], dtype=np.float64)
                vectors = np.asarray(arrays['features'], dtype=np.float64)
        except (OSError, KeyError, TypeError) as error:
            raise ValueError(f'Missing or unreadable feature cache for group {gid}') from error
        if (times.ndim != 1 or len(times) == 0 or vectors.shape != (len(times), FEATURE_DIM)
                or not np.isfinite(times).all() or not np.isfinite(vectors).all()
                or (times < 0).any() or (np.diff(times) <= 0).any()):
            raise ValueError('Invalid feature vectors or timestamps')
        group = dict(group_id=gid, row_ids=[r['row_id'] for r in source_rows],
                     positive=bool(types), training_type_ids=types, anchors=anchors,
                     table_type_sets_conflict=len({tuple(sorted(r['table_type_ids'])) for r in source_rows}) > 1,
                     timestamps=times, features=vectors)
        if any(not np.any(label_mask(group, t) == 1) for t in types):
            raise ValueError('No cached frames fall within a positive source anchor')
        groups.append(group)
    return dict(source_manifest=manifest, rows=rows, groups=sorted(groups, key=lambda g: g['group_id']),
                classes=[dict(type=t, title=TITLES[t]) for t in classes],
                feature_version=FEATURE_VERSION, backbone_sha256=backbone_sha,
                manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cache_metadata_sha256=hashlib.sha256((cache / 'metadata.json').read_bytes()).hexdigest())
