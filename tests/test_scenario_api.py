"""Scenario contracts and real API lifecycle with inference replaced at its boundary."""
import copy
import importlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.settings import Settings, get_settings
from src.vision.adapter import AnalysisError, normalize_result, run_analysis

api = importlib.import_module('src.api.app')
adapter = importlib.import_module('src.vision.adapter')


def scenario_result(candidate=True):
    _, encoded = cv2.imencode('.jpg', np.zeros((24, 32, 3), np.uint8))
    return {
        'analysis_scope': 'bank_scenarios',
        'verdict': 'review_required' if candidate else 'no_candidate',
        'events': [{
            'event_type': 'safe_left_open', 'timecode_sec': .5,
            'start_sec': 0., 'end_sec': 1., 'confidence': .8,
            'description': 'Scored window; review context.',
            'evidence_frames': [{'timestamp_sec': .5, 'jpeg_bytes': encoded.tobytes()}],
        }] if candidate else [],
        'model_version': 'scenario-test-v1', 'feature_version': 'mobilenetv2-gap-temporal-v1',
        'score_semantics': 'uncalibrated_model_score',
        'summary': {'sample_count': 2, 'sample_fps': 1., 'events_truncated': False},
        'class_results': [
            {'type': 'safe_left_open', 'title': 'Safe left open', 'score': .8 if candidate else .2,
             'threshold': .65, 'decision': 'candidate' if candidate else 'not_flagged',
             'positive_groups': 2, 'negative_groups': 3, 'peak_sec': .5},
            {'type': 'identity_check_missing', 'title': 'Document check not observed', 'score': None,
             'threshold': .65, 'decision': 'unsupported', 'positive_groups': 0,
             'negative_groups': 3, 'peak_sec': None},
        ],
        'limitations': ['Experimental candidates; no procedural conclusion.'],
    }


@pytest.mark.parametrize('candidate', [True, False])
def test_scenario_metadata_survives_normalization_and_public_response(candidate):
    normalized = normalize_result(scenario_result(candidate), 2., {})
    public = api._public_result('job', json.dumps(normalized), 0)
    assert public['analysis_scope'] == 'bank_scenarios'
    assert public['verdict'] == ('review_required' if candidate else 'no_candidate')
    assert public['class_results'][0]['decision'] == ('candidate' if candidate else 'not_flagged')
    assert public['class_results'][1]['score'] is None
    assert public['class_results'][1]['peak_sec'] is None
    assert public['feature_version'] == 'mobilenetv2-gap-temporal-v1'
    assert public['score_semantics'] == 'uncalibrated_model_score'
    assert public['limitations'] == ['Experimental candidates; no procedural conclusion.']
    assert public['summary']['sample_count'] == 2
    if candidate:
        assert public['violations'][0]['start_sec'] == 0.
        assert public['violations'][0]['end_sec'] == 1.
        assert public['violations'][0]['confidence'] == .8
        assert public['violations'][0]['evidence_frames'][0]['image_base64'].startswith('/9j/')
    else:
        assert public['violations'] == []


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(verdict='no_violation'),
    lambda r: r.update(verdict={}),
    lambda r: r.update(verdict='no_candidate'),
    lambda r: r.update(events=[]),
    lambda r: r.update(class_results=[]),
    lambda r: r['class_results'].append(copy.deepcopy(r['class_results'][0])),
    lambda r: r['class_results'][0].update(type='unknown_procedure'),
    lambda r: r['class_results'][0].update(decision='violation'),
    lambda r: r['class_results'][0].update(decision=[]),
    lambda r: r['class_results'][0].update(score=float('nan')),
    lambda r: r['class_results'][0].update(score=float('inf')),
    lambda r: r['class_results'][0].update(score=True),
    lambda r: r['class_results'][0].update(score=.1),
    lambda r: r['class_results'][0].update(threshold=float('nan')),
    lambda r: r['class_results'][0].update(threshold=1.1),
    lambda r: r['class_results'][0].update(positive_groups=0),
    lambda r: r['class_results'][0].update(positive_groups=-1),
    lambda r: r['class_results'][0].update(negative_groups=True),
    lambda r: r['class_results'][0].update(negative_groups=1.5),
    lambda r: r['class_results'][0].update(peak_sec=3),
    lambda r: r['class_results'][0].update(peak_sec=None),
    lambda r: r['class_results'][1].update(score=0.),
    lambda r: r['class_results'][1].update(peak_sec=0.),
    lambda r: r['class_results'][1].update(positive_groups=2),
    lambda r: r['events'][0].update(event_type='identity_check_missing'),
    lambda r: r['events'][0].update(start_sec=1.5),
    lambda r: r['events'][0].update(confidence=float('nan')),
    lambda r: r['events'][0].update(evidence_frames=[]),
    lambda r: r['summary'].update(sample_count=True),
    lambda r: r['summary'].update(sample_count=1201),
    lambda r: r['summary'].update(sample_count=0),
    lambda r: r['summary'].update(sample_fps=0),
    lambda r: r['summary'].update(sample_fps=float('nan')),
    lambda r: r['summary'].update(events_truncated='false'),
    lambda r: r.update(score_semantics='probability_correct'),
    lambda r: r.update(model_version=''),
    lambda r: r.update(feature_version='x' * 4001),
    lambda r: r.update(limitations=['']),
    lambda r: r.update(limitations=['note'] * 51),
    lambda r: r.update(limitations='not a list'),
    lambda r: r.update(events=r['events'] * 101),
    lambda r: r.update(class_results=r['class_results'] * 26),
])
def test_scenario_rejects_malformed_or_contradictory_output(mutate):
    raw = scenario_result()
    mutate(raw)
    with pytest.raises(AnalysisError):
        normalize_result(raw, 2., {})


def settings(tmp_path, **changes):
    values = dict(api_key='test-key', upload_dir=tmp_path / 'uploads',
                  db_path=tmp_path / 'jobs.sqlite3', max_upload_mb=1,
                  analyzer_mode='real', analyzer_callable='src.vision.safe_classifier:analyze_video',
                  model_path='safe-weights', rules_path=None)
    values.update(changes)
    return Settings(**values)


def post(client, **data):
    return client.post('/api/v1/analyze', headers={'X-API-Key': 'test-key'},
                       files={'file': ('clip.mov', b'video-bytes')}, data=data)


def wait_result(client, job_id):
    for _ in range(150):
        result = client.get('/api/v1/jobs/' + job_id, headers={'X-API-Key': 'test-key'}).json()
        if result['status'] != 'processing':
            return result
        time.sleep(.01)
    pytest.fail('Job did not finish')


@pytest.mark.parametrize('mode', ['other', 'src.vision.evil:run', '', 'BANK_SCENARIOS'])
def test_unknown_analysis_modes_are_rejected_before_upload(tmp_path, mode):
    with TestClient(api.create_app(settings(tmp_path))) as client:
        response = post(client, analysis_type=mode, roi='[0,0,1,1]')
        assert response.status_code == 422
    assert not list((tmp_path / 'uploads').glob('*.mov*'))


@pytest.mark.parametrize('mode', ['safe_state', 'bank_scenarios'])
def test_explicit_mode_is_rejected_in_demo(tmp_path, mode):
    with TestClient(api.create_app(settings(tmp_path, analyzer_mode='demo'))) as client:
        assert post(client, analysis_type=mode, roi='[0,0,1,1]').status_code == 422


def test_explicit_safe_mode_requires_roi_when_default_is_another_analyzer(tmp_path):
    with TestClient(api.create_app(settings(tmp_path, analyzer_callable='tests.fake_analyzers:no_violation'))) as client:
        assert post(client, analysis_type='safe_state').status_code == 422


@pytest.mark.parametrize('outcome', ['done', 'failure', 'timeout'])
def test_selected_scenario_skips_safe_roi_and_cleans_private_sidecar(tmp_path, monkeypatch, outcome):
    def infer(path, configured, timeout_sec, stop_event=None):
        selection = json.loads(Path(str(path) + '.analysis.json').read_text())
        assert selection == {'analysis_type': 'bank_scenarios'}
        assert not Path(str(path) + '.roi.json').exists()
        if outcome == 'failure':
            raise ValueError('private model path')
        if outcome == 'timeout':
            raise TimeoutError('private timeout')
        return normalize_result(scenario_result(False), 2, {})
    monkeypatch.setattr(api, '_run_child', infer)
    with TestClient(api.create_app(settings(tmp_path))) as client:
        accepted = post(client, analysis_type='bank_scenarios')
        assert accepted.status_code == 202
        result = wait_result(client, accepted.json()['job_id'])
        assert result['status'] == ('done' if outcome == 'done' else 'failed')
        if outcome == 'done':
            assert result['verdict'] == 'no_candidate'
        else:
            assert 'private' not in result['error']
    assert not list((tmp_path / 'uploads').glob('*.mov*'))


def test_restart_removes_analysis_sidecar_for_interrupted_job(tmp_path):
    configured = settings(tmp_path)
    api._initialize(configured)
    path = configured.upload_dir / 'interrupted.mov'
    path.write_bytes(b'movie')
    Path(str(path) + '.roi.json').write_text('{}')
    Path(str(path) + '.analysis.json').write_text('{"analysis_type":"bank_scenarios"}')
    with api._connect(configured) as db:
        db.execute('INSERT INTO jobs(job_id,status,video_path,created_at,updated_at) VALUES(?,?,?,?,?)',
                   ('interrupted', 'processing', str(path), time.time(), time.time()))
    api._initialize(configured)
    assert not list(configured.upload_dir.glob('*.mov*'))


def test_sidecar_write_failure_cleans_uploaded_video(tmp_path, monkeypatch):
    original = Path.write_text
    def fail_selection(self, *args, **kwargs):
        if str(self).endswith('.analysis.json'):
            raise OSError('simulated storage error')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', fail_selection)
    with TestClient(api.create_app(settings(tmp_path)), raise_server_exceptions=False) as client:
        assert post(client, analysis_type='bank_scenarios').status_code == 500
    assert not list((tmp_path / 'uploads').glob('*.mov*'))


@pytest.mark.parametrize('mode,expected_path', [('bank_scenarios', 'scenario-weights'), ('safe_state', 'safe-weights')])
def test_adapter_dispatches_selected_mode_using_only_server_paths(tmp_path, monkeypatch, mode, expected_path):
    movie = tmp_path / 'clip.mov'
    movie.write_bytes(b'fake')
    Path(str(movie) + '.analysis.json').write_text(json.dumps({'analysis_type': mode}))
    monkeypatch.setattr(adapter, 'inspect_video', lambda p: (2., b'jpeg'))
    expected_module = 'src.vision.scenario_classifier' if mode == 'bank_scenarios' else 'src.vision.safe_classifier'
    def import_analyzer(module):
        assert module == expected_module
        def infer(path):
            assert path == str(movie)
            assert os.environ['MODEL_PATH'] == expected_path
            return scenario_result(False) if mode == 'bank_scenarios' else {'verdict': 'no_violation', 'events': []}
        return SimpleNamespace(analyze_video=infer)
    monkeypatch.setattr(adapter.importlib, 'import_module', import_analyzer)
    monkeypatch.setenv('MODEL_PATH', 'untrusted-stale-path')
    result = run_analysis(str(movie), settings(tmp_path, scenario_model_path='scenario-weights'))
    assert result['verdict'] == ('no_candidate' if mode == 'bank_scenarios' else 'no_violation')


@pytest.mark.parametrize('selection', ['bad json', '{}', '{"analysis_type":"other"}',
                                       '{"analysis_type":"bank_scenarios","model_path":"outside"}'])
def test_adapter_rejects_invalid_selection_sidecar(tmp_path, monkeypatch, selection):
    movie = tmp_path / 'clip.mov'
    Path(str(movie) + '.analysis.json').write_text(selection)
    monkeypatch.setattr(adapter, 'inspect_video', lambda p: (2., b'jpeg'))
    # The configured fallback is valid; an invalid explicit choice must not silently use it.
    monkeypatch.setattr(adapter.importlib, 'import_module', lambda name: SimpleNamespace(
        analyze_video=lambda path: {'verdict': 'no_violation', 'events': []}))
    with pytest.raises(AnalysisError):
        run_analysis(str(movie), settings(tmp_path))


def test_scenario_path_comes_from_its_own_environment_setting(tmp_path, monkeypatch):
    monkeypatch.setenv('API_KEY', 'test-key')
    monkeypatch.setenv('MODEL_PATH', 'safe-weights')
    monkeypatch.setenv('SCENARIO_MODEL_PATH', 'scenario-weights')
    configured = get_settings()
    assert configured.model_path == 'safe-weights'
    assert configured.scenario_model_path == 'scenario-weights'


def test_bad_sidecar_encoding_is_a_fixed_analysis_error(tmp_path):
    movie = tmp_path / 'clip.mov'
    Path(str(movie) + '.analysis.json').write_bytes(b'\xff\xfe')
    with pytest.raises(AnalysisError):
        run_analysis(str(movie), settings(tmp_path))


def test_missing_scenario_model_cannot_reuse_the_safe_model(tmp_path, monkeypatch):
    movie = tmp_path / 'clip.mov'
    Path(str(movie) + '.analysis.json').write_text('{"analysis_type":"bank_scenarios"}')
    monkeypatch.setenv('MODEL_PATH', 'safe-weights')
    with pytest.raises(AnalysisError, match='not configured'):
        run_analysis(str(movie), settings(tmp_path))


def test_omitted_selection_preserves_configured_analyzer(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, 'inspect_video', lambda path: (2., b'jpeg'))
    result = run_analysis(str(tmp_path / 'clip.mov'), settings(
        tmp_path, analyzer_callable='tests.fake_analyzers:no_violation'))
    assert result == {'verdict': 'no_violation', 'violations': [], 'demo': False}


def test_expiry_removes_private_selection_and_roi(tmp_path):
    configured = settings(tmp_path, job_ttl_sec=1)
    api._initialize(configured)
    movie = configured.upload_dir / 'expired.mov'
    for suffix in ('', '.analysis.json', '.roi.json'):
        Path(str(movie) + suffix).write_bytes(b'old')
    with api._connect(configured) as db:
        db.execute('INSERT INTO jobs(job_id,status,video_path,created_at,updated_at) VALUES(?,?,?,?,?)',
                   ('expired', 'done', str(movie), time.time() - 2, time.time() - 2))
    api._cleanup_expired(configured)
    assert not list(configured.upload_dir.glob('*.mov*'))
