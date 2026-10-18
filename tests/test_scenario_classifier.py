import hashlib
import json

import numpy as np
import pytest


def make_model(tmp_path):
    from src.vision.scenario_features import BACKBONE_SHA256, FEATURE_VERSION
    coef = np.zeros((2, 768))
    coef[0, 0] = 1.
    np.savez(tmp_path / 'model.npz', coef=coef, intercept=np.zeros(2), mean=np.zeros(768),
             scale=np.ones(768), supported=np.array([True, False]))
    metadata = dict(schema_version=1, feature_version=FEATURE_VERSION, model_version='scenario-test', threshold=.65,
                    backbone_sha256=BACKBONE_SHA256,
                    artifact_sha256=hashlib.sha256((tmp_path / 'model.npz').read_bytes()).hexdigest(),
                    classes=[dict(type='safe_left_open', title='Сейф оставлен открытым', positive_groups=2, negative_groups=4),
                             dict(type='identity_check_missing', title='Проверка документа', positive_groups=0, negative_groups=4)],
                    training_scope='all_available_weak_labels', limitations=['Weak labels'])
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    return tmp_path


def test_scores_have_no_negative_decision_for_an_untrained_class(tmp_path):
    from src.vision.scenario_classifier import load_model, score_features
    model = load_model(make_model(tmp_path))
    vectors = np.zeros((4, 768))
    vectors[:, 0] = [-2, 1, 2, -2]
    results, scores = score_features(model, np.array([0., 1., 2., 3.]), vectors)
    assert results[0]['decision'] == 'candidate'
    assert results[0]['peak_sec'] == 2.
    assert results[0]['score'] == pytest.approx(.8807970779)
    assert results[1]['decision'] == 'unsupported'
    assert results[1]['score'] is None and results[1]['peak_sec'] is None
    assert np.isnan(scores[:, 1]).all()


def test_candidate_window_stops_at_below_threshold_sample_and_real_gap():
    from src.vision.scenario_classifier import peak_interval
    assert peak_interval(np.array([0., 1., 2., 3., 4.]), np.array([.2, .7, .9, .8, .2]), .65) == (1, 2, 3)
    assert peak_interval(np.array([0., 1., 20.]), np.array([.7, .9, .8]), .65) == (0, 1, 1)


def test_tampered_numeric_artifact_is_rejected(tmp_path):
    from src.vision.scenario_classifier import load_model
    path = make_model(tmp_path)
    (path / 'model.npz').write_bytes(b'not the fitted artifact')
    with pytest.raises(ValueError):
        load_model(path)


def test_v2_support_policy_abstains_on_singleton_while_legacy_remains_loadable(tmp_path):
    from src.vision.scenario_classifier import load_model, score_features
    from src.vision.scenario_features import RESIDUAL_FEATURE_VERSION
    path = make_model(tmp_path)
    metadata_path = path / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(schema_version=2, feature_version=RESIDUAL_FEATURE_VERSION,
                    support_policy=dict(min_positive_media_groups=2, min_negative_media_groups=1))
    metadata['classes'][1]['positive_groups'] = 1
    metadata_path.write_text(json.dumps(metadata))
    model = load_model(path)
    scores, _ = score_features(model, np.arange(2.), np.zeros((2, 768)))
    assert scores[1]['decision'] == 'unsupported' and scores[1]['score'] is None
    metadata['support_policy']['min_positive_media_groups'] = True
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_model(path)


def test_v2_singleton_abstention_survives_service_normalization():
    from tests.test_scenario_api import scenario_result
    from src.vision.adapter import normalize_result, AnalysisError
    raw = scenario_result(False)
    raw['model_support_policy'] = dict(min_positive_media_groups=2, min_negative_media_groups=1)
    raw['class_results'][1]['positive_groups'] = 1
    normalized = normalize_result(raw, 2., {})
    assert normalized['class_results'][1]['decision'] == 'unsupported'
    assert normalized['model_support_policy']['min_positive_media_groups'] == 2
    raw['model_support_policy']['min_positive_media_groups'] = True
    with pytest.raises(AnalysisError):
        normalize_result(raw, 2., {})


def test_inference_is_pixel_based_and_outputs_candidate_window_evidence(tmp_path, monkeypatch):
    import cv2
    from src.vision import scenario_classifier as module
    model_dir = make_model(tmp_path)
    monkeypatch.setenv('MODEL_PATH', str(model_dir))
    video = tmp_path / 'anonymous.mov'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 2, (64, 48))
    for _ in range(8):
        writer.write(np.full((48, 64, 3), 80, np.uint8))
    writer.release()
    vectors = np.zeros((4, 768))
    vectors[:, 0] = [-2, 1, 2, -2]
    # Only external backbone extraction is substituted; scoring, intervals,
    # real media decoding and evidence encoding are exercised.
    monkeypatch.setattr(module, 'extract_video_features', lambda path, backbone: (np.arange(4.), vectors))
    result = module.analyze_video(str(video))
    assert result['analysis_scope'] == 'bank_scenarios'
    assert result['verdict'] == 'review_required'
    assert len(result['events']) == 1
    event = result['events'][0]
    assert (event['start_sec'], event['timecode_sec'], event['end_sec']) == (1., 2., 2.)
    assert [frame['timestamp_sec'] for frame in event['evidence_frames']] == [1., 2.]
    assert all(frame['jpeg_bytes'].startswith(b'\xff\xd8') for frame in event['evidence_frames'])
    assert 'окно' in event['description'].lower()
    assert result['class_results'][1]['decision'] == 'unsupported'
