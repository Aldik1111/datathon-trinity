"""Coverage is established by completed decoding, never by a declared duration."""
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

from src.vision import video_io


def make_video(path):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for index in range(23):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 8, np.uint8), format='bgr24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_complete_decode_records_actual_pts_and_tail(tmp_path):
    path = tmp_path / 'synthetic.mov'
    make_video(path)
    coverage = {}
    samples = list(video_io.iter_sampled_frames(path, 1., coverage=coverage))
    assert coverage['full_decode'] is True
    assert coverage['decoded_frame_count'] == 23
    assert coverage['sample_count'] == 3
    assert coverage['sample_timestamps_sec'] == [time for time, _ in samples] == [0., 1., 2.]
    assert coverage['last_frame_sec'] == pytest.approx(2.2)
    assert coverage['duration_sec'] == pytest.approx(2.3)
    assert coverage['max_frame_gap_sec'] == pytest.approx(.1)


def test_abandoned_iterator_cannot_claim_full_decode(tmp_path):
    path = tmp_path / 'synthetic.mov'
    make_video(path)
    coverage = {'full_decode': True, 'stale': 'must be cleared'}
    iterator = video_io.iter_sampled_frames(path, 1., coverage=coverage)
    next(iterator)
    iterator.close()
    assert coverage == {'full_decode': False}


@pytest.mark.parametrize('declared,points', [(4, [0, 1, 2]), (3, [0, 2, 1])])
def test_short_decode_and_broken_pts_do_not_publish_coverage(monkeypatch, declared, points):
    stream = SimpleNamespace(frames=declared, average_rate=10, metadata={})

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, selected):
            for pts in points:
                yield SimpleNamespace(pts=pts, time_base=Fraction(1, 10), duration=1,
                                      rotation=0, to_ndarray=lambda **kwargs: np.zeros((48, 64, 3), np.uint8))

    monkeypatch.setattr(video_io.av, 'open', lambda _: Container())
    coverage = {}
    with pytest.raises(ValueError):
        list(video_io.iter_sampled_frames('synthetic.mov', 1., coverage=coverage))
    assert coverage == {'full_decode': False}
