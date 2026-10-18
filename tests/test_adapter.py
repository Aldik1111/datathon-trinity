import base64
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.vision.adapter import AnalysisError, load_rules, normalize_result, run_analysis
from src.vision.contracts import AnalysisResult, Event, EvidenceFrame


def jpeg():
    ok, encoded = cv2.imencode('.jpg', np.zeros((24, 32, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def event(time=0.5):
    return Event('test', time, 'Visible action', [EvidenceFrame(time, jpeg())])


def test_multiple_events_and_jpeg_round_trip():
    output = normalize_result(AnalysisResult('violation', [event(0.5), event(1.5)]), 2, {})
    assert len(output['violations']) == 2
    assert output['violations'][0]['rule_status'] == 'unverified'
    image = base64.b64decode(output['violations'][0]['evidence_frames'][0]['image_base64'])
    assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape == (24, 32, 3)


@pytest.mark.parametrize('raw', [AnalysisResult('violation', []), AnalysisResult('no_violation', [event()])])
def test_reject_inconsistent_verdict(raw):
    with pytest.raises(AnalysisError):
        normalize_result(raw, 2, {})


@pytest.mark.parametrize('timestamp', [-1, float('nan'), float('inf'), 2.001, 3, True, '0.5'])
def test_reject_invalid_timestamps(timestamp):
    with pytest.raises(AnalysisError):
        normalize_result(AnalysisResult('violation', [event(timestamp)]), 2, {})


@pytest.mark.parametrize('frames', [[], [EvidenceFrame(0, b'bad')], [EvidenceFrame(0, jpeg())] * 4])
def test_reject_invalid_evidence(frames):
    with pytest.raises(AnalysisError):
        normalize_result(AnalysisResult('violation', [Event('test', 0, 'text', frames)]), 2, {})


def test_no_violation_is_empty():
    assert normalize_result(AnalysisResult('no_violation'), 2, {}) == {'verdict': 'no_violation', 'violations': []}


@pytest.mark.parametrize('duration', [0, -1, True, float('nan'), float('inf'), None])
def test_no_negative_verdict_without_valid_video_duration(duration):
    with pytest.raises(AnalysisError):
        normalize_result(AnalysisResult('no_violation'), duration, {})


def test_only_verified_rules_are_exposed(tmp_path):
    import json
    rule = dict(rule_id='sample', title='Test source', url='https://example.org/rule', section='1', checked_at='2026-10-17', limitations='Test fixture only', verified=True)
    registry = tmp_path / 'rules.json'
    registry.write_text(json.dumps([rule, dict(rule, rule_id='unverified', verified=False)]))
    rules = load_rules(str(registry))
    assert set(rules) == {'sample'}
    item = event()
    item.rule_id = 'sample'
    output = normalize_result(AnalysisResult('violation', [item]), 2, rules)
    assert output['violations'][0]['rule']['url'] == rule['url']


def test_corrupt_video_fails_even_in_demo(tmp_path):
    path = tmp_path / 'bad.mov'
    path.write_bytes(b'not a movie')
    with pytest.raises(AnalysisError):
        run_analysis(str(path), SimpleNamespace(analyzer_mode='demo'))
