"""Derive a separate offline scene-residual cache without rereading private video."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.vision.scenario_features import FEATURE_VERSION, RESIDUAL_FEATURE_VERSION, scene_residual_features


def derive(source, destination):
    source, destination = Path(source), Path(destination)
    metadata_path = source / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Expected original temporal features')
    destination.mkdir(parents=True, exist_ok=False)
    metadata.update(feature_version=RESIDUAL_FEATURE_VERSION,
                    transform='per_clip_median_center; no fitted parameters or labels',
                    source_metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest())
    for path in sorted(source.glob('*.npz')):
        with np.load(path, allow_pickle=False) as arrays:
            np.savez_compressed(destination / path.name, timestamps=arrays['timestamps'],
                                features=scene_residual_features(arrays['features']))
    (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    derive(args.source, args.output)
