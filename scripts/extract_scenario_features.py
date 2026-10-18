"""Extract all local originals once; cache identity uses media bytes and feature version."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from src.vision.scenario_features import BACKBONE_SHA256, FEATURE_VERSION, FrameEncoder, extract_video_features


def extract(manifest_path, backbone_path, output_dir):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(feature_version=FEATURE_VERSION, backbone_sha256=BACKBONE_SHA256, sample_fps=1,
                    manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest())
    meta_path = output / 'metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text(encoding='utf-8')) != metadata:
        raise ValueError('Feature cache belongs to different data or preprocessing; choose a new directory')
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    cv2.setNumThreads(4)
    encoder = FrameEncoder(backbone_path)
    total = 0
    for group in manifest['groups']:
        original = Path(group['local_original_path'])
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if digest != group['group_id']:
            raise ValueError('Original media checksum changed')
        destination = output / (digest + '.npz')
        started = time.perf_counter()
        if destination.exists():
            with np.load(destination, allow_pickle=False) as saved:
                times, vectors = saved['timestamps'], saved['features']
                if vectors.shape != (len(times), 768) or not np.isfinite(vectors).all():
                    raise ValueError('Invalid cached features')
        else:
            times, vectors = extract_video_features(original, backbone_path, encoder=encoder)
            np.savez_compressed(destination, timestamps=times, features=vectors)
        total += len(times)
        print(json.dumps(dict(rows=group['row_ids'], samples=len(times), seconds=round(time.perf_counter()-started, 2)), ensure_ascii=False), flush=True)
    print(json.dumps(dict(groups=len(manifest['groups']), samples=total)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    extract(args.manifest, args.backbone, args.output)
