"""Read-only annotation/video export with static calibration and group isolation.

No inferred/interpolated labels are introduced. The split is fixed from source
groups before learning, and identical media is represented by one canonical copy.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_roi(roi):
    try:
        x, y, w, h = map(float, roi)
        return all(math.isfinite(v) for v in (x, y, w, h)) and (
            x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= 1 + 1e-8 and y + h <= 1 + 1e-8)
    except (TypeError, ValueError):
        return False


def _roi(box):
    return [box.get(k) for k in ("x", "y", "width", "height")]


def _state(box):
    """Prefer the attribute schema; accept legacy state only when absent."""
    return box.get("attributes", {}).get("state", box.get("state"))


def eligible_boxes(video):
    """Return clear independently annotated open/closed boxes plus audit reasons."""
    keep, excluded = [], []
    for box in video.get("annotations", {}).get("boxes", []):
        if box.get("label") != "safe":
            continue
        attributes = box.get("attributes", {})
        reason = None
        if box.get("generated") or box.get("interpolated") or box.get("source_keyframes"):
            reason = "generated"
        elif attributes.get("visibility") != "clear":
            reason = "visibility_not_clear"
        elif _state(box) not in ("open", "closed"):
            reason = "ambiguous_state"
        elif not valid_roi(_roi(box)):
            reason = "invalid_roi"
        else:
            try:
                timestamp = float(box["time_sec"])
                if not math.isfinite(timestamp) or timestamp < 0:
                    reason = "invalid_timestamp"
            except (ValueError, TypeError, KeyError):
                reason = "invalid_timestamp"
        if reason:
            excluded.append(dict(video_id=video["id"], annotation_id=box.get("id"), reason=reason))
        else:
            keep.append(box)
    return keep, excluded


def union_roi(boxes):
    rois = [_roi(box) for box in boxes]
    if not rois or not all(valid_roi(roi) for roi in rois):
        raise ValueError("Static calibration needs valid normalized safe boxes")
    x, y = min(r[0] for r in rois), min(r[1] for r in rois)
    right, bottom = max(r[0] + r[2] for r in rois), max(r[1] + r[3] for r in rois)
    return [x, y, min(1., right) - x, min(1., bottom) - y]


def oriented_rgb(frame, rotation=None):
    """PyAV display-matrix rotation is counterclockwise, matching annotation UI."""
    angle = int(round(frame.rotation if rotation is None else rotation))
    if angle % 90:
        raise ValueError(f"Unsupported non-right-angle display rotation: {angle}")
    rgb = frame.to_ndarray(format="rgb24")
    return np.ascontiguousarray(np.rot90(rgb, (angle // 90) % 4))


def crop_rgb(image, roi):
    if not valid_roi(roi):
        raise ValueError("Invalid normalized ROI")
    height, width = image.shape[:2]
    x, y, w, h = roi
    left, top = math.floor(x * width + 1e-8), math.floor(y * height + 1e-8)
    right, bottom = min(width, math.ceil((x + w) * width - 1e-8)), min(height, math.ceil((y + h) * height - 1e-8))
    if right <= left or bottom <= top:
        raise ValueError("ROI produces an empty crop")
    return image[top:bottom, left:right].copy()


def extract_frames(path, timestamps):
    """First frame at/after each exact annotation time (within 1 ms rounding).

    Annotation time remains in the manifest; actual relative PTS is separate.
    Reject out-of-range targets rather than silently reusing the last frame.
    """
    pending = sorted(set(timestamps))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else None
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                continue
            absolute = float(frame.pts * frame.time_base)
            if origin is None:
                origin = absolute
            actual = absolute - origin
            if pending and actual >= pending[0] - .001:
                rgb = oriented_rgb(frame)
                while pending and actual >= pending[0] - .001:
                    target = pending.pop(0)
                    if actual - target > .15:
                        raise ValueError(f"Target {target} has no nearby frame in {path}")
                    yield target, actual, rgb, int(frame.rotation)
            if not pending:
                break
    if pending:
        raise ValueError(f"Annotation times outside video: {path}: {pending}")


def _number(video):
    match = re.search(r"-(\d+)\.mov$", video["filename"], flags=re.I)
    if not match:
        raise ValueError(f"Cannot map filename to original media: {video['filename']}")
    return int(match[1])


def assign_group_splits(groups):
    """Deterministic 60/20/20 group split, balancing counts without model output.

    Try fixed-seed partitions and choose nearest sample/class proportions while
    requiring both labels in every partition. No frames cross group boundaries.
    """
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("At least three independent video groups required")
    n_val = max(1, round(len(keys) * .2))
    n_test = max(1, round(len(keys) * .2))
    n_train = len(keys) - n_val - n_test
    totals = Counter(label for items in groups.values() for label in items)
    ratios = {"train": .6, "val": .2, "test": .2}
    rng = random.Random(20261018)
    best = None
    for _ in range(5000):
        shuffled = keys.copy()
        rng.shuffle(shuffled)
        mapping = {key: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
                   for i, key in enumerate(shuffled)}
        counts = {split: Counter(label for key in keys if mapping[key] == split for label in groups[key])
                  for split in ratios}
        if any(counts[s][label] == 0 for s in ratios for label in ("open", "closed")):
            continue
        score = sum(((counts[s][label] - totals[label] * ratios[s]) / max(1, totals[label])) ** 2
                    for s in ratios for label in ("open", "closed"))
        tie = tuple(mapping[k] for k in keys)
        if best is None or (score, tie) < best[:2]:
            best = score, tie, mapping
    if best is None:
        raise ValueError("Cannot place open and closed labels in all group-isolated splits")
    return best[2]


def contact_sheets(manifest, output):
    for split in ("train", "val", "test"):
        samples = [s for s in manifest["samples"] if s["split"] == split]
        for offset in range(0, len(samples), 24):
            subset = samples[offset:offset + 24]
            sheet = Image.new("RGB", (960, math.ceil(len(subset) / 4) * 150), "#ededed")
            draw = ImageDraw.Draw(sheet)
            for i, sample in enumerate(subset):
                x, y = i % 4 * 240, i // 4 * 150
                image = Image.open(sample["image_path"]).convert("RGB")
                image.thumbnail((232, 120))
                sheet.paste(image, (x + 4, y + 26))
                number = re.search(r"-(\d+)\.mov$", sample["filename"], re.I)[1]
                draw.text((x + 4, y + 5), f"X-{number} {sample['timestamp_sec']:g}s {sample['label']}", fill="black")
            sheet.save(output / f"contact-{split}-{offset // 24 + 1}.jpg", quality=95)


def export_dataset(snapshot_path, media_root, output_dir):
    snapshot_path, media_root, output = Path(snapshot_path), Path(media_root), Path(output_dir).resolve()
    source = json.loads(snapshot_path.read_text(encoding="utf-8"))
    candidates, exclusions = [], []
    # Filter before mapping names: unrelated uploads need no original media.
    selected = []
    for video in source["videos"]:
        boxes, rejected = eligible_boxes(video)
        exclusions.extend(rejected)
        if boxes:
            selected.append((video, boxes))
    for video, boxes in sorted(selected, key=lambda item: _number(item[0])):
        path = (media_root / f"X-{_number(video)}" / "media.mov").resolve()
        group = sha256_file(path)
        # Include every independently annotated valid safe box in calibration,
        # including partial views: this is geometry, never a training label.
        calibration = [b for b in video["annotations"]["boxes"] if b.get("label") == "safe"
                       and not b.get("generated") and not b.get("interpolated")
                       and not b.get("source_keyframes") and valid_roi(_roi(b))]
        candidates.append(dict(video=video, boxes=boxes, calibration=calibration, path=path, group=group))
    by_group = defaultdict(list)
    for candidate in candidates:
        by_group[candidate["group"]].append(candidate)
    group_labels = {g: [_state(b) for b in copies[0]["boxes"]] for g, copies in by_group.items()}
    splits = assign_group_splits(group_labels)
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    manifest = dict(schema_version=1, samples=[], videos=[], exclusions=exclusions,
                    source_sha256=sha256_file(snapshot_path), split_seed=20261018,
                    split_method="sha256 media groups; fixed-seed 60/20/20 class-balanced candidates",
                    calibration="static union of independently annotated safe boxes per media group",
                    limitations=["Same-camera evaluation; group split does not establish unseen-camera generalization.",
                                 "Sparse frame labels do not establish event interval ground truth."])
    for group, copies in by_group.items():
        canonical = copies[0]
        roi = union_roi([b for c in copies for b in c["calibration"]])
        for copy in copies:
            video = copy["video"]
            manifest["videos"].append(dict(video_id=video["id"], filename=video["filename"],
                path=str(copy["path"]), group_id=group, split=splits[group], roi=roi,
                canonical_video_id=canonical["video"]["id"]))
            if copy is not canonical:
                for box in copy["boxes"]:
                    exclusions.append(dict(video_id=video["id"], annotation_id=box.get("id"),
                        reason="duplicate_video", canonical_video_id=canonical["video"]["id"]))
        boxes_by_time = defaultdict(list)
        for box in canonical["boxes"]:
            boxes_by_time[float(box["time_sec"])].append(box)
        for target, actual, image, rotation in extract_frames(canonical["path"], boxes_by_time):
            same_time = boxes_by_time[target]
            labels = {_state(b) for b in same_time}
            if len(labels) != 1:
                raise ValueError(f"Conflicting safe labels at {target}: {canonical['path']}")
            box = same_time[0]
            video = canonical["video"]
            path = output / "images" / f"x{_number(video):02d}-{target:.6f}.png"
            Image.fromarray(crop_rgb(image, roi)).save(path)
            manifest["samples"].append(dict(image_path=str(path), label=next(iter(labels)), split=splits[group],
                video_id=video["id"], filename=video["filename"], group_id=group, timestamp_sec=target,
                decoded_timestamp_sec=actual, roi=roi, annotation_roi=_roi(box), annotation_id=box.get("id"),
                rotation_deg=rotation))
            for duplicate in same_time[1:]:
                exclusions.append(dict(video_id=video["id"], annotation_id=duplicate.get("id"), reason="duplicate_timestamp"))
    manifest["summary"] = {split: dict(samples=sum(s["split"] == split for s in manifest["samples"]),
        labels=dict(Counter(s["label"] for s in manifest["samples"] if s["split"] == split)),
        groups=sum(s == split for s in splits.values()),
        videos=sum(v["split"] == split for v in manifest["videos"])) for split in ("train", "val", "test")}
    manifest["exclusion_counts"] = dict(Counter(e["reason"] for e in exclusions))
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheets(manifest, output)
    return manifest
