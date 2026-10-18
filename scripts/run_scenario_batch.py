"""Sequential real HTTP run of all table videos; compare with frozen final-head scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ui.client import APIClientError, get_job, load_settings, submit_video


def reusable_job(saved, media_sha, model_version, settings):
    result = saved.get('result', {})
    if not (saved.get('source_media_sha256') == media_sha and result.get('status') == 'done'
            and result.get('model_version') == model_version and result.get('job_id')):
        return None
    try:
        live = get_job(settings, result['job_id'])
    except APIClientError as exc:
        if 'Ошибка API (404):' in str(exc):
            return None
        raise
    if (live.get('status') != 'done' or live.get('model_version') != model_version
            or live.get('analysis_scope') != 'bank_scenarios' or live.get('demo') is not False):
        return None
    return live


def run(manifest_path, comparison_path, output_dir):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    comparison_path = Path(comparison_path)
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    by_row = {row['row_id']: row for row in comparison['rows']}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    summary = []
    for source in manifest['rows']:
        started = time.monotonic()
        row_id = source['row_id']
        target = output / f'{row_id}.json'
        original = Path(source['original_media']['path'])
        media = original.read_bytes()
        media_sha = hashlib.sha256(media).hexdigest()
        if media_sha != source['group_id']:
            raise ValueError('Source media changed before HTTP comparison')
        saved = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
        result = reusable_job(saved, media_sha, comparison['model_version'], settings)
        if result is None:
            pending = submit_video(settings, source['filename'], media, analysis_type='bank_scenarios')
            print(json.dumps(dict(row=row_id, job_id=pending['job_id'], status='processing')), flush=True)
            deadline = time.monotonic() + 305
            while True:
                result = get_job(settings, pending['job_id'])
                if result['status'] != 'processing' or time.monotonic() >= deadline:
                    break
                time.sleep(.5)
            target.write_text(json.dumps(dict(row_id=row_id, filename=source['filename'],
                                             source_media_sha256=media_sha, result=result),
                                         ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        matched = False
        if result.get('status') == 'done':
            if (result.get('demo') is not False or result.get('analysis_scope') != 'bank_scenarios'
                    or result.get('model_version') != comparison['model_version']):
                raise ValueError('HTTP response used the wrong model or mode')
            expected = {r['type']: r for r in by_row[row_id]['in_sample']['class_results']}
            actual = {r['type']: r for r in result['class_results']}
            matched = actual.keys() == expected.keys() and all(
                actual[key]['decision'] == value['decision'] and
                (actual[key]['score'] is None and value['score'] is None or
                 actual[key]['score'] is not None and value['score'] is not None and
                 abs(actual[key]['score'] - value['score']) < 1e-8)
                for key, value in expected.items())
            by_row[row_id]['final_job_id'] = result['job_id']
        item = dict(row_id=row_id, filename=source['filename'], job_id=result.get('job_id'), status=result.get('status'),
                    seconds=round(time.monotonic()-started, 2), final_scores_match=matched,
                    candidate_types=[r['type'] for r in result.get('class_results', []) if r['decision'] == 'candidate'])
        if result.get('error'):
            item['error'] = result['error']
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        (output / 'http-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    comparison['http_verification'] = dict(rows=len(summary), completed=sum(r['status'] == 'done' for r in summary),
                                           matched=sum(r['final_scores_match'] for r in summary))
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if not all(row['status'] == 'done' and row['final_scores_match'] for row in summary):
        raise RuntimeError('One or more HTTP results failed or disagreed with frozen model scores')
    print(json.dumps(comparison['http_verification']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--comparison', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.manifest, args.comparison, args.output)
