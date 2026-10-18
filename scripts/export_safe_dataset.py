"""Export original annotated safe crops; never writes original media or annotations."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.dataset import export_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/safe-dataset"))
    args = parser.parse_args()
    manifest = export_dataset(args.snapshot, args.media_root, args.output)
    print(json.dumps({"manifest": str((args.output / "manifest.json").resolve()),
                      "summary": manifest["summary"], "exclusions": manifest["exclusion_counts"]}, indent=2))
