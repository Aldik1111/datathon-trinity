"""Decode real PTS, group state runs, and retain actual image evidence."""
import importlib
import json
from fractions import Fraction

import av
import cv2
import numpy as np
import pytest


def api():
    return importlib.import_module('src.vision.safe_classifier')


def test_roi_validation_rejects_missing_nonfinite_and_outside(tmp_path, monkeypatch):
    classifier = api()
    video = tmp_path / 'video.mp4'
    with pytest.raises(ValueError, match='ROI'):
        classifier.analyze_video(str(video))
    for roi in ([0, 0, 2, 1], [0, 0, 0, 1], [0, 0, float('nan'), 1], [True, 0, 1, 1]):
        (tmp_path / 'video.mp4.roi.json').write_text(json.dumps(dict(roi=roi)))
        with pytest.raises(ValueError, match='ROI'):
            classifier.analyze_video(str(video))


def test_contiguous_states_and_closed_timeline_have_real_evidence(tmp_path, monkeypatch):
    classifier = api()
    video = tmp_path / 'different-name.mov'
    (tmp_path / 'different-name.mov.roi.json').write_text(json.dumps(dict(roi=[.1, .2, .5, .6])))
    states = ['closed', 'open', 'open', 'unknown', 'closed', 'open']
    monkeypatch.setenv('MODEL_PATH', str(tmp_path))
    monkeypatch.setattr(classifier, 'load_model', lambda _: type('Model', (), {'metadata': {'model_version': 'fixture'}})())
    monkeypatch.setattr(classifier, 'iter_sampled_frames', lambda _: iter((i * .5, np.full((100, 160, 3), i * 30, np.uint8)) for i in range(6)))
    predictions = iter(states)
    monkeypatch.setattr(classifier, 'predict_crop', lambda model, crop: dict(state=next(predictions), confidence=.9, open_probability=.9))
    result = classifier.analyze_video(str(video))
    assert result['verdict'] == 'review_required'
    assert [observation['state'] for observation in result['observations']] == states
    assert [(event['start_sec'], event['end_sec']) for event in result['events']] == [(.5, 1.0), (1.5, 1.5), (2.5, 2.5)]
    evidence = result['events'][0]['evidence_frames']
    decoded = cv2.imdecode(np.frombuffer(evidence[0]['jpeg_bytes'], np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (100, 160)
    assert 20 < decoded[0, 0].mean() < 40
    assert np.max(decoded[20, 16]) > 100  # visible ROI overlay
    assert len(evidence) <= 3


def test_closed_only_means_no_event_and_many_runs_are_reported_as_truncated(tmp_path, monkeypatch):
    classifier = api()
    video = tmp_path / 'video.mov'
    (tmp_path / 'video.mov.roi.json').write_text(json.dumps(dict(roi=[0, 0, 1, 1])))
    monkeypatch.setenv('MODEL_PATH', str(tmp_path))
    monkeypatch.setattr(classifier, 'load_model', lambda _: type('Model', (), {'metadata': {'model_version': 'fixture'}})())
    frame = np.full((64, 64, 3), 30, np.uint8)
    monkeypatch.setattr(classifier, 'iter_sampled_frames', lambda _: iter([(0., frame), (.5, frame)]))
    monkeypatch.setattr(classifier, 'predict_crop', lambda *args: dict(state='closed', confidence=.9, open_probability=.1))
    assert classifier.analyze_video(str(video))['verdict'] == 'no_event'
    monkeypatch.setattr(classifier, 'iter_sampled_frames', lambda _: iter((i * .5, frame) for i in range(220)))
    counter = iter('open' if i % 2 else 'closed' for i in range(220))
    monkeypatch.setattr(classifier, 'predict_crop', lambda *args: dict(state=next(counter), confidence=.9, open_probability=.9))
    result = classifier.analyze_video(str(video))
    assert len(result['events']) == 100
    assert result['summary']['events_truncated'] is True
    assert len(result['observations']) == 220


def test_decode_sampling_uses_relative_pts_and_keeps_end_of_stream(tmp_path):
    video_io = importlib.import_module('src.vision.video_io')
    path = tmp_path / 'timestamps.mkv'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 32, 'bgr0'
        stream.time_base = Fraction(1, 10)
        for index in range(13):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), index * 10, np.uint8), format='bgr24')
            frame.pts, frame.time_base = index + 20, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames = list(video_io.iter_sampled_frames(path, 2))
    assert [round(timestamp, 2) for timestamp, frame in frames] == [0, .5, 1.0]
    assert int(frames[-1][1].mean()) == 100


def test_display_rotation_preserves_pixel_direction():
    video_io = importlib.import_module('src.vision.video_io')
    image = np.zeros((20, 40, 3), np.uint8)
    image[:10, :20] = 255
    # AV display matrix rotation is counterclockwise.
    rotated = video_io.apply_rotation(image, 90)
    assert rotated.shape == (40, 20, 3)
    assert rotated[-1, 0].mean() == 255
    assert rotated[0, 0].mean() == 0


def test_mov_display_matrix_is_applied_during_real_decode(tmp_path):
    video_io = importlib.import_module('src.vision.video_io')
    path = tmp_path / 'rotated.mov'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=2)
        stream.width, stream.height, stream.pix_fmt = 64, 32, 'yuv420p'
        stream.set_display_rotation(90)
        image = np.zeros((32, 64, 3), np.uint8)
        image[:16, :32] = 255
        for index in range(2):
            frame = av.VideoFrame.from_ndarray(image, format='bgr24')
            frame.pts, frame.time_base = index, Fraction(1, 2)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    timestamp, decoded = next(video_io.iter_sampled_frames(path))
    assert timestamp == 0
    assert decoded.shape == (64, 32, 3)
    assert decoded[-4, 4].mean() > 240
    assert decoded[4, 4].mean() < 10


def test_late_decode_failure_never_returns_partial_analysis(tmp_path, monkeypatch):
    classifier = api()
    video = tmp_path / 'video.mov'
    (tmp_path / 'video.mov.roi.json').write_text(json.dumps(dict(roi=[0, 0, 1, 1])))
    monkeypatch.setenv('MODEL_PATH', str(tmp_path))
    monkeypatch.setattr(classifier, 'load_model', lambda _: type('Model', (), {'metadata': {'model_version': 'fixture'}})())
    monkeypatch.setattr(classifier, 'predict_crop', lambda *args: dict(state='closed', confidence=.9, open_probability=.1))
    def frames(_):
        yield 0., np.zeros((64, 64, 3), np.uint8)
        raise ValueError('Late decode failure')
    monkeypatch.setattr(classifier, 'iter_sampled_frames', frames)
    with pytest.raises(ValueError, match='Late decode'):
        classifier.analyze_video(str(video))
