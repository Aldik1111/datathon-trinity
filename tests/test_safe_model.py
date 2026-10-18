"""Learn from images, preserve numeric artifacts, and prevent split leakage."""
import importlib
import hashlib
import json

import cv2
import numpy as np
import pytest


def modules():
    return importlib.import_module('src.vision.safe_model'), importlib.import_module('src.training.train_safe')


def make_manifest(tmp_path):
    samples = []
    for split in ('train', 'val', 'test'):
        for label, color in [('open', (20, 25, 210)), ('closed', (200, 180, 25))]:
            for number in range(3):
                image = np.full((64, 64, 3), color, np.uint8)
                cv2.rectangle(image, (10 + number, 8), (40, 55), (10, 10, 10), 3)
                path = tmp_path / f'{split}-{label}-{number}.png'
                cv2.imwrite(str(path), image)
                samples.append(dict(image_path=str(path), label=label, split=split,
                                    group_id=f'{split}-{label}', video_id=f'{split}-{label}',
                                    filename=path.name, timestamp_sec=number, roi=[0, 0, 1, 1]))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, samples=samples, videos=[], source_sha256='fixture')))
    return manifest


def test_training_roundtrip_uses_pixels_and_numeric_artifacts(tmp_path):
    model_api, training = modules()
    manifest = make_manifest(tmp_path)
    output = tmp_path / 'model'
    report = training.train(manifest, output)
    model = model_api.load_model(output)
    for label in ('open', 'closed'):
        crop = cv2.imread(str(tmp_path / f'test-{label}-0.png'))
        prediction = model_api.predict_crop(model, crop)
        assert prediction['state'] == label
        assert 0 <= prediction['open_probability'] <= 1
        assert prediction == model_api.predict_crop(model_api.load_model(output), crop.copy())
    with np.load(output / 'model.npz', allow_pickle=False) as artifact:
        assert all(artifact[key].dtype.kind == 'f' for key in artifact.files)
    assert report['splits']['test']['samples'] == 6
    assert report['splits']['test']['videos'] == 2
    assert 'confusion_matrix' in report['splits']['test']
    assert report['score_semantics'] == 'uncalibrated_model_score'
    again = tmp_path / 'model2'
    training.train(manifest, again)
    assert model_api.load_model(again).metadata['model_version'] == model.metadata['model_version']


def test_training_rejects_group_leakage_before_writing_artifacts(tmp_path):
    _, training = modules()
    path = make_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest['samples'][-1]['group_id'] = manifest['samples'][0]['group_id']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='split'):
        training.train(path, tmp_path / 'model')
    assert not (tmp_path / 'model' / 'model.npz').exists()


def test_invalid_crops_abstain_and_zero_margin_abstains(tmp_path):
    model_api, training = modules()
    training.train(make_manifest(tmp_path), tmp_path / 'model')
    model = model_api.load_model(tmp_path / 'model')
    for crop in (None, np.zeros((2, 2, 3), np.uint8), np.zeros((32, 800, 3), np.uint8),
                 np.zeros((32, 32), np.uint8), np.full((32, 32, 3), np.nan)):
        prediction = model_api.predict_crop(model, crop)
        assert prediction['state'] == 'unknown'
        assert prediction['confidence'] == 0
    model.coef[:] = 0
    model.intercept[:] = 0
    assert model_api.predict_crop(model, np.zeros((64, 64, 3), np.uint8))['state'] == 'unknown'


def test_artifact_tampering_and_nonfinite_weights_are_rejected(tmp_path):
    model_api, training = modules()
    output = tmp_path / 'model'
    training.train(make_manifest(tmp_path), output)
    (output / 'model.npz').write_bytes(b'not an npz')
    with pytest.raises(ValueError, match='artifact'):
        model_api.load_model(output)


def test_numeric_integrity_is_checked_even_with_matching_checksum(tmp_path):
    model_api, training = modules()
    output = tmp_path / 'model'
    training.train(make_manifest(tmp_path), output)
    with np.load(output / 'model.npz', allow_pickle=False) as arrays:
        values = {key: arrays[key].copy() for key in arrays.files}
    values['coef'][0] = np.nan
    np.savez(output / 'model.npz', **values)
    metadata = json.loads((output / 'metadata.json').read_text())
    metadata['artifact_sha256'] = hashlib.sha256((output / 'model.npz').read_bytes()).hexdigest()
    (output / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='artifact'):
        model_api.load_model(output)


def test_test_pixels_and_labels_never_select_weights_or_threshold(tmp_path):
    _, training = modules()
    path = make_manifest(tmp_path)
    first = tmp_path / 'first'
    training.train(path, first)
    manifest = json.loads(path.read_text())
    for sample in manifest['samples']:
        if sample['split'] == 'test':
            sample['label'] = 'closed' if sample['label'] == 'open' else 'open'
            cv2.imwrite(sample['image_path'], np.full((64, 64, 3), (12, 7, 6), np.uint8))
    path.write_text(json.dumps(manifest))
    second = tmp_path / 'second'
    training.train(path, second)
    before = json.loads((first / 'metadata.json').read_text())
    after = json.loads((second / 'metadata.json').read_text())
    assert before['model_version'] == after['model_version']
    assert before['threshold'] == after['threshold']
    with np.load(first / 'model.npz') as left, np.load(second / 'model.npz') as right:
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])


def test_features_are_finite_and_content_dependent():
    model_api, _ = modules()
    dark = np.full((64, 64, 3), 30, np.uint8)
    bright = np.full((64, 64, 3), 200, np.uint8)
    assert np.isfinite(model_api.features(dark)).all()
    assert not np.array_equal(model_api.features(dark), model_api.features(bright))
