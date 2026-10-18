"""Dataset safeguards: provenance, group isolation, orientation, and static crops."""
import json
import shutil
from fractions import Fraction

import av
import numpy as np
import pytest
from PIL import Image


def box(identifier="a", state="closed", time=0, **updates):
    value = dict(id=identifier, label="safe", time_sec=time, x=.25, y=.25,
                 width=.5, height=.5, attributes=dict(state=state, visibility="clear"))
    value.update(updates)
    return value


def test_filter_drops_generated_ambiguous_and_invalid_boxes():
    from src.training.dataset import eligible_boxes
    video = dict(id="v", annotations=dict(boxes=[
        box(), box("generated", generated="interpolation"),
        box("unclear", attributes=dict(state="open", visibility="partial")),
        box("unknown", attributes=dict(state="unknown", visibility="clear")),
        box("nan", x=float("nan")), box("outside", x=.9),
        box("negative_time", time=-1), box("person", label="person"),
    ]))
    keep, excluded = eligible_boxes(video)
    assert [b["id"] for b in keep] == ["a"]
    assert {e["reason"] for e in excluded} == {
        "generated", "visibility_not_clear", "ambiguous_state", "invalid_roi", "invalid_timestamp"}


def test_static_roi_encloses_open_door_and_has_no_label_dependent_crop():
    from src.training.dataset import union_roi, crop_rgb
    roi = union_roi([box(x=.3, width=.3), box(state="open", x=.1, width=.5)])
    assert roi == pytest.approx([.1, .25, .5, .5])
    image = np.zeros((40, 100, 3), dtype=np.uint8)
    assert crop_rgb(image, roi).shape == (20, 50, 3)
    with pytest.raises(ValueError):
        crop_rgb(image, [.9, 0, .2, .5])


def test_display_orientation_rotates_real_frame_counterclockwise():
    from src.training.dataset import oriented_rgb
    raw = np.zeros((16, 32, 3), dtype=np.uint8)
    raw[:, :16] = [255, 0, 0]
    raw[:, 16:] = [0, 0, 255]
    frame = av.VideoFrame.from_ndarray(raw, format="rgb24")
    result = oriented_rgb(frame, rotation=90)
    assert result.shape == (32, 16, 3)
    assert result[0, 0].tolist() == [0, 0, 255]
    assert result[-1, 0].tolist() == [255, 0, 0]


def make_video(path, color):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = 32, 16, "yuv420p"
        for i in range(4):
            pixels = np.full((16, 32, 3), color + i * 10, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 2)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("state_location", ["attributes", "top_level"])
def test_export_preserves_annotation_time_deduplicates_and_groups_splits(tmp_path, state_location):
    from src.training.dataset import export_dataset
    media = tmp_path / "media"
    media.mkdir()
    videos = []
    for i in range(1, 7):
        directory = media / f"X-{i}"
        directory.mkdir()
        make_video(directory / "media.mov", i * 20)
        videos.append(dict(id=f"v{i}", filename=f"X-{i}.mov", annotations=dict(boxes=[
            box(f"c{i}", time=0), box(f"o{i}", state="open", time=.49, x=.1, width=.65)])))
        for annotation in videos[-1]["annotations"]["boxes"]:
            if state_location == "top_level":
                annotation["state"] = annotation["attributes"].pop("state")
            else:
                # A present attributes.state takes precedence over legacy state.
                annotation["state"] = "unknown"
    duplicate = media / "X-7"
    duplicate.mkdir()
    shutil.copyfile(media / "X-1/media.mov", duplicate / "media.mov")
    videos.append(dict(id="v7", filename="X-7.mov", annotations=videos[0]["annotations"]))
    # Unrelated annotation entries need no local media or numeric filename.
    videos.append(dict(id="unused", filename="unrelated-copy.mov", annotations=dict(boxes=[])))
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(dict(videos=videos)), encoding="utf-8")
    first = export_dataset(snapshot, media, tmp_path / "out")
    second = export_dataset(snapshot, media, tmp_path / "out2")
    assert len(first["samples"]) == 12
    assert first["source_sha256"] == second["source_sha256"]
    assert [(s["video_id"], s["split"]) for s in first["samples"]] == [
        (s["video_id"], s["split"]) for s in second["samples"]]
    for split in ("train", "val", "test"):
        assert {s["label"] for s in first["samples"] if s["split"] == split} == {"closed", "open"}
    for group in {s["group_id"] for s in first["videos"]}:
        assert len({v["split"] for v in first["videos"] if v["group_id"] == group}) == 1
    assert any(e["reason"] == "duplicate_video" and e["video_id"] == "v7" for e in first["exclusions"])
    opened = next(s for s in first["samples"] if s["label"] == "open")
    assert opened["timestamp_sec"] == .49
    assert opened["decoded_timestamp_sec"] == .5
    assert opened["roi"] == pytest.approx([.1, .25, .65, .5])
    assert Image.open(opened["image_path"]).mode == "RGB"
    assert (tmp_path / "out/manifest.json").exists()
