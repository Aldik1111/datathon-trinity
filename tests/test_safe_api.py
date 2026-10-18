import importlib
import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.settings import Settings
from src.vision.adapter import AnalysisError, normalize_result

api = importlib.import_module('src.api.app')


def safe_result(state='open'):
    _, encoded = cv2.imencode('.jpg', np.zeros((24, 32, 3), np.uint8))
    return {
        'verdict': 'no_event' if state == 'closed' else 'review_required',
        'events': [] if state == 'closed' else [{
            'event_type': 'candidate_safe_open', 'timecode_sec': 0.5,
            'start_sec': 0.5, 'end_sec': 1.5, 'confidence': 0.8,
            'description': 'Open state requires context.',
            'evidence_frames': [{'timestamp_sec': 0.5, 'jpeg_bytes': encoded.tobytes()}],
        }],
        'analysis_scope': 'safe_state', 'model_version': 'safe-v1',
        'summary': 'Safe state only.',
        'observations': [{'timecode_sec': 0.5, 'state': state, 'confidence': 0.8}],
    }


def test_scoped_result_retains_evidence_intervals_and_observations():
    normalized = normalize_result(safe_result(), 2, {})
    public = api._public_result('id', json.dumps(normalized), 0)
    assert public['verdict'] == 'review_required'
    assert public['analysis_scope'] == 'safe_state'
    assert public['observations'] == [{'timecode_sec': 0.5, 'state': 'open', 'confidence': 0.8}]
    assert public['violations'][0]['end_sec'] == 1.5
    assert public['violations'][0]['confidence'] == 0.8
    assert public['demo'] is False


def test_closed_safe_result_never_becomes_no_violation():
    normalized = normalize_result(safe_result('closed'), 2, {})
    assert normalized['verdict'] == 'no_event'
    assert normalized['violations'] == []


def test_structured_summary_matches_the_observations():
    raw = safe_result()
    raw['summary'] = {'sample_count': 1, 'state_counts': {'open': 1, 'closed': 0, 'unknown': 0},
                      'sample_fps': 2, 'events_truncated': False,
                      'score_semantics': 'uncalibrated_model_score'}
    normalized = normalize_result(raw, 2, {})
    assert normalized['summary']['state_counts']['open'] == 1
    raw['summary']['state_counts']['closed'] = 5
    with pytest.raises(AnalysisError):
        normalize_result(raw, 2, {})


@pytest.mark.parametrize('mutate', [
    lambda r: r.pop('analysis_scope'),
    lambda r: r.update(verdict='no_violation'),
    lambda r: r['observations'][0].update(confidence=float('nan')),
    lambda r: r['observations'][0].update(timecode_sec=3),
    lambda r: r['observations'][0].update(state='locked'),
    lambda r: r['events'][0].update(start_sec=1.5, end_sec=0.5),
    lambda r: r.update(verdict='no_event'),
])
def test_scoped_result_rejects_invalid_semantics(mutate):
    raw = safe_result()
    mutate(raw)
    with pytest.raises(AnalysisError):
        normalize_result(raw, 2, {})


def settings(tmp_path):
    return Settings(api_key='test-key', upload_dir=tmp_path / 'uploads',
                    db_path=tmp_path / 'jobs.sqlite3', max_upload_mb=1,
                    analyzer_mode='real', analyzer_callable='src.vision.safe_classifier:analyze_video')


@pytest.mark.parametrize('roi', ['[0,0,0,1]', '[0.9,0,0.2,1]', '[0,0,NaN,1]', '[true,0,1,1]', '{}', 'bad', None])
def test_invalid_or_missing_roi_rejected_before_queue(tmp_path, roi):
    with TestClient(api.create_app(settings(tmp_path))) as client:
        response = client.post('/api/v1/analyze', headers={'X-API-Key': 'test-key'},
                               files={'file': ('x.mov', b'video')},
                               data={} if roi is None else {'roi': roi})
        assert response.status_code == 422
        assert not list((tmp_path / 'uploads').glob('*.mov*'))


@pytest.mark.parametrize('fail', [False, True])
def test_roi_reaches_worker_and_sidecar_is_cleaned(tmp_path, monkeypatch, fail):
    seen = []
    def analyze(path, settings, timeout_sec, stop_event=None):
        from pathlib import Path
        seen.append(json.loads(Path(str(path) + '.roi.json').read_text()))
        if fail:
            raise ValueError('simulated failure')
        return normalize_result(safe_result('closed'), 2, {})
    monkeypatch.setattr(api, '_run_child', analyze)
    with TestClient(api.create_app(settings(tmp_path))) as client:
        response = client.post('/api/v1/analyze', headers={'X-API-Key': 'test-key'},
                               files={'file': ('x.mov', b'video')},
                               data={'roi': '[0.1,0.2,0.3,0.4]'})
        assert response.status_code == 202
        job_id = response.json()['job_id']
        for _ in range(100):
            result = client.get('/api/v1/jobs/' + job_id, headers={'X-API-Key': 'test-key'}).json()
            if result['status'] != 'processing':
                break
            time.sleep(0.02)
        assert result['status'] == ('failed' if fail else 'done')
    assert seen == [{'roi': [0.1, 0.2, 0.3, 0.4]}]
    assert not list((tmp_path / 'uploads').glob('*.mov*'))
