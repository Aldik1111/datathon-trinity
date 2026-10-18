"""The jury boundary preserves model policy and refuses misleading negatives."""
import base64
from copy import deepcopy

import cv2
import numpy as np
import pytest

from src.vision.adapter import AnalysisError
from src.vision.jury_projection import project_scenario_result
from src.vision.scenario_contracts import SCENARIO_TYPES
from src.vision.scenario_features import FEATURE_VERSION


def normalized_result(candidate=True):
    ok, image = cv2.imencode('.jpg', np.full((24, 32, 3), 64, np.uint8))
    assert ok
    encoded = base64.b64encode(image.tobytes()).decode('ascii')
    classes = [dict(type=kind, title=kind, score=.2, threshold=.65,
                    decision='not_flagged', positive_groups=2, negative_groups=3,
                    peak_sec=1.) for kind in sorted(SCENARIO_TYPES)]
    if candidate:
        row = next(item for item in classes if item['type'] == 'safe_left_open')
        row.update(score=.8, decision='candidate')
    return dict(
        analysis_scope='bank_scenarios', verdict='review_required' if candidate else 'no_candidate',
        model_version='scenario-test-v1', feature_version=FEATURE_VERSION,
        model_support_policy=dict(min_positive_media_groups=1, min_negative_media_groups=1),
        score_semantics='uncalibrated_model_score', class_results=classes,
        limitations=['Weakly supervised candidates require human review.'],
        summary=dict(sample_count=3, sample_fps=1., events_truncated=False),
        coverage=dict(full_decode=True, decoded_frame_count=6, sample_count=3,
                      sample_fps=1., first_sample_sec=0., last_sample_sec=2.,
                      last_frame_sec=2.5, duration_sec=3., max_frame_gap_sec=.5,
                      sample_timestamps_sec=[0., 1., 2.]),
        violations=[dict(type='safe_left_open', timecode_sec=1., start_sec=0., end_sec=2.,
                         confidence=.8, description='Untrusted claim: someone committed a violation.',
                         rule=None, rule_status='unverified',
                         evidence_frames=[dict(timestamp_sec=t, image_base64=encoded) for t in (0., 1., 2.)])]
        if candidate else [],
    )


@pytest.mark.parametrize('candidate', [True, False])
def test_jury_binary_projection_preserves_internal_evidence_and_model_policy(candidate):
    source = normalized_result(candidate)
    before = deepcopy(source)
    result = project_scenario_result(source, 3.)
    assert result['verdict'] == ('violation' if candidate else 'no_violation')
    assert result['internal_verdict'] == ('review_required' if candidate else 'no_candidate')
    assert result['output_contract'] == 'jury-v1'
    assert result['jury_policy'] == 'scenario-threshold-v1'
    assert result['class_results'] == source['class_results']
    assert result['model_support_policy'] == source['model_support_policy']
    assert result['coverage'] == source['coverage']
    assert result['model_version'] == source['model_version']
    assert result['feature_version'] == source['feature_version']
    assert result['analysis_scope'] == 'bank_scenarios'
    assert result['score_semantics'] == 'uncalibrated_model_score'
    assert source == before
    assert result['rules_status'] == 'unverified'
    assert 'Бинарное решение' in result['limitations'][-1]
    if candidate:
        event = result['violations'][0]
        assert 'Untrusted claim' not in event['description']
        assert 'не является вероятностью' in event['description']
        assert event['evidence_frames'] == source['violations'][0]['evidence_frames']
        for frame in event['evidence_frames']:
            pixels = cv2.imdecode(np.frombuffer(base64.b64decode(frame['image_base64'], validate=True), np.uint8), cv2.IMREAD_COLOR)
            assert pixels.shape == (24, 32, 3)
    else:
        assert result['violations'] == []


def make_unsupported(source):
    source['class_results'][0].update(positive_groups=0, decision='unsupported', score=None, peak_sec=None)


def test_positive_candidate_does_not_claim_unsupported_classes_were_checked():
    source = normalized_result()
    make_unsupported(source)
    result = project_scenario_result(source, 3.)
    assert result['verdict'] == 'violation'
    assert result['class_results'][0]['decision'] == 'unsupported'
    assert source['class_results'][0]['type'] in result['limitations'][-1]
    assert 'Не оценены классы' in result['limitations'][-1]


def test_unsupported_class_never_becomes_a_binary_negative():
    source = normalized_result(False)
    make_unsupported(source)
    with pytest.raises(AnalysisError, match='unsupported'):
        project_scenario_result(source, 3.)


def test_candidate_at_saved_threshold_remains_eligible_without_new_cutoff():
    source = normalized_result()
    row = next(item for item in source['class_results'] if item['type'] == 'safe_left_open')
    row.update(threshold=.73, score=.73)
    source['violations'][0]['confidence'] = .73
    result = project_scenario_result(source, 3.)
    assert result['verdict'] == 'violation'
    assert result['violations'][0]['confidence'] == .73
    assert next(item for item in result['class_results'] if item['type'] == row['type'])['threshold'] == .73


@pytest.mark.parametrize('duration', [0., -1., float('nan'), float('inf'), True, '3'])
def test_invalid_duration_cannot_produce_empty_success(duration):
    with pytest.raises(AnalysisError, match='duration'):
        project_scenario_result(normalized_result(False), duration)


@pytest.mark.parametrize('mutate', [
    lambda r: r.pop('coverage'),
    lambda r: r['coverage'].update(full_decode=False),
    lambda r: r['coverage'].update(full_decode=1),
    lambda r: r['coverage'].update(decoded_frame_count=True),
    lambda r: r['coverage'].update(decoded_frame_count=2),
    lambda r: r['coverage'].update(sample_count=2),
    lambda r: r['coverage'].update(sample_fps=2.),
    lambda r: r['coverage'].update(first_sample_sec=.1),
    lambda r: r['coverage'].update(last_sample_sec=1.),
    lambda r: r['coverage'].update(last_frame_sec=1.5),
    lambda r: r['coverage'].update(duration_sec=4.),
    lambda r: r['coverage'].update(max_frame_gap_sec=2.),
    lambda r: r['coverage'].update(max_frame_gap_sec=float('nan')),
    lambda r: r['coverage'].update(sample_timestamps_sec=[0., 2., 1.]),
    lambda r: r['coverage'].update(sample_timestamps_sec=[0., 1., 1.]),
    lambda r: r['coverage'].update(sample_timestamps_sec=[0., 1., float('inf')]),
    lambda r: r['summary'].update(events_truncated=True),
    lambda r: r['summary'].update(sample_count=2),
    lambda r: r['summary'].update(sample_fps=2.),
])
def test_incomplete_or_inconsistent_coverage_fails_even_with_a_candidate(mutate):
    source = normalized_result()
    mutate(source)
    with pytest.raises(AnalysisError):
        project_scenario_result(source, 3.)


def test_unobserved_long_tail_cannot_become_a_negative_decision():
    source = normalized_result(False)
    source['coverage']['duration_sec'] = 10.
    with pytest.raises(AnalysisError, match='coverage'):
        project_scenario_result(source, 10.)


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(analysis_scope='safe_state'),
    lambda r: r.update(verdict='violation'),
    lambda r: r.update(verdict='no_candidate'),
    lambda r: r.update(score_semantics='probability_of_violation'),
    lambda r: r.update(feature_version='unknown'),
    lambda r: r.update(model_version=''),
    lambda r: r.update(class_results=r['class_results'][:-1]),
    lambda r: r['class_results'].__setitem__(0, deepcopy(r['class_results'][1])),
    lambda r: r['class_results'][0].update(type='unknown'),
    lambda r: r['class_results'][0].update(score=float('nan')),
    lambda r: r['class_results'][0].update(threshold=float('inf')),
    lambda r: r['class_results'][0].update(decision='candidate'),
    lambda r: r['class_results'][0].update(positive_groups=0),
    lambda r: r['model_support_policy'].update(min_positive_media_groups=True),
    lambda r: r.update(violations=[]),
    lambda r: r.update(violations=r['violations'] * 2),
    lambda r: r['violations'][0].update(type='identity_check_missing'),
    lambda r: r['violations'][0].update(confidence=.7),
    lambda r: r['violations'][0].update(timecode_sec=2.),
    lambda r: r['violations'][0].update(start_sec=2.),
    lambda r: r['violations'][0].update(end_sec=3.01),
    lambda r: r['violations'][0].update(timecode_sec=float('nan')),
    lambda r: r['violations'][0].update(timecode_sec=True),
    lambda r: r['violations'][0].update(start_sec=.1),
    lambda r: r.update(limitations=[]),
])
def test_projection_rejects_inconsistent_candidates_and_metadata(mutate):
    source = normalized_result()
    mutate(source)
    with pytest.raises(AnalysisError):
        project_scenario_result(source, 3.)


@pytest.mark.parametrize('mutate', [
    lambda e: e.update(evidence_frames=[]),
    lambda e: e.update(evidence_frames=e['evidence_frames'] + [deepcopy(e['evidence_frames'][0])]),
    lambda e: e.update(evidence_frames=[deepcopy(e['evidence_frames'][0])] * 2),
    lambda e: e.update(evidence_frames=[e['evidence_frames'][0], e['evidence_frames'][2]]),
    lambda e: e['evidence_frames'][0].update(timestamp_sec=-.1),
    lambda e: e['evidence_frames'][0].update(timestamp_sec=3.01),
    lambda e: e['evidence_frames'][0].update(timestamp_sec=.5),
    lambda e: e['evidence_frames'][0].update(timestamp_sec=float('inf')),
    lambda e: e['evidence_frames'][0].update(image_base64='%%%'),
    lambda e: e['evidence_frames'][0].update(image_base64=base64.b64encode(b'not JPEG').decode()),
    lambda e: e['evidence_frames'][0].update(image_base64=base64.b64encode(b'\xff\xd8broken\xff\xd9').decode()),
])
def test_jury_evidence_requires_real_decodable_jpeg_at_analyzed_event_samples(mutate):
    source = normalized_result()
    mutate(source['violations'][0])
    with pytest.raises(AnalysisError):
        project_scenario_result(source, 3.)


def test_valid_jpeg_from_another_interval_is_rejected():
    source = normalized_result()
    source['violations'][0].update(start_sec=1., end_sec=2.)
    with pytest.raises(AnalysisError, match='inside the event interval'):
        project_scenario_result(source, 3.)


def test_one_peak_jpeg_is_enough_and_verified_rule_metadata_survives():
    source = normalized_result()
    event = source['violations'][0]
    event['evidence_frames'] = [event['evidence_frames'][1]]
    event.update(rule_status='verified', rule={'rule_id': 'verified-test-rule'})
    result = project_scenario_result(source, 3.)
    assert len(result['violations'][0]['evidence_frames']) == 1
    assert result['violations'][0]['rule'] == {'rule_id': 'verified-test-rule'}
    assert result['rules_status'] == 'verified_sources_present'


def test_jpeg_decoder_exception_becomes_a_fixed_safe_analysis_error(monkeypatch):
    from src.vision import jury_projection
    source = normalized_result()
    def fail_decoder(*args):
        raise cv2.error('private implementation details')
    monkeypatch.setattr(jury_projection.cv2, 'imdecode', fail_decoder)
    with pytest.raises(AnalysisError, match='cannot be decoded') as error:
        project_scenario_result(source, 3.)
    assert 'private' not in str(error.value)
