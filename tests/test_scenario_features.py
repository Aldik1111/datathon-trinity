import numpy as np
import pytest


def test_letterbox_preserves_whole_frame_colors_and_zero_normalized_padding():
    from src.vision.scenario_features import preprocess_frame
    frame = np.zeros((112, 224, 3), np.uint8)
    frame[:, :112] = (0, 0, 255)
    frame[:, 112:] = (255, 0, 0)
    blob = preprocess_frame(frame)
    assert blob.shape == (1, 3, 224, 224)
    assert np.allclose(blob[0, :, 0, 100], [0, 0, 0])
    assert np.allclose(blob[0, :, 60, 10], [(1-.485)/.229, -.456/.224, -.406/.225], atol=1e-5)
    assert np.allclose(blob[0, :, 60, 210], [-.485/.229, -.456/.224, (1-.406)/.225], atol=1e-5)


def test_temporal_context_uses_actual_seconds_and_direction_not_absolute_time():
    from src.vision.scenario_features import temporal_features
    times = np.array([0., 1., 2., 3., 4.])
    embeddings = np.arange(5, dtype=float).reshape(-1, 1)
    out = temporal_features(times, embeddings)
    assert out.shape == (5, 3)
    assert np.allclose(out[2], [2., 2., 3.])
    assert np.allclose(out[0], [0., 1., 1.5])
    assert np.allclose(out, temporal_features(times + 100, embeddings))
    # A discontinuity in actual time must not become an adjacent context frame.
    assert np.allclose(temporal_features(np.array([0., 10.]), np.array([[1.], [9.]])), [[1., 1., 0.], [9., 9., 0.]])


@pytest.mark.parametrize('times,vectors', [([0, 0], [[1], [2]]), ([1, 0], [[1], [2]]), ([0], [[float('nan')]]), ([0, 1], [[1]])])
def test_invalid_temporal_inputs_fail_instead_of_publishing_partial_result(times, vectors):
    from src.vision.scenario_features import temporal_features
    with pytest.raises(ValueError):
        temporal_features(np.asarray(times), np.asarray(vectors))


def test_renaming_video_does_not_change_extracted_features(tmp_path):
    import cv2
    from src.vision.scenario_features import extract_video_features
    first = tmp_path / 'X-1.mov'
    writer = cv2.VideoWriter(str(first), cv2.VideoWriter_fourcc(*'mp4v'), 2, (64, 48))
    for level in (10, 40, 80, 120, 160, 200):
        writer.write(np.full((48, 64, 3), level, np.uint8))
    writer.release()
    renamed = tmp_path / 'unseen-upload.mov'
    renamed.write_bytes(first.read_bytes())

    class PixelEncoder:
        def encode(self, frame):
            return np.full(256, frame.mean() / 255, dtype=np.float32)

    ta, xa = extract_video_features(first, None, encoder=PixelEncoder())
    tb, xb = extract_video_features(renamed, None, encoder=PixelEncoder())
    assert np.allclose(ta, [0, 1, 2])
    assert xa.shape == (3, 768)
    assert np.array_equal(ta, tb)
    assert np.array_equal(xa, xb)
