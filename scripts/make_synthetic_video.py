"""Create only artificial geometry; never read any source video or archive."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/synthetic.mov"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("MOV encoder is unavailable")
    try:
        for index in range(20):
            frame = np.full((240, 320, 3), (30, 40, 50), dtype=np.uint8)
            cv2.rectangle(frame, (10 + index * 8, 80), (50 + index * 8, 120), (20, 200, 230), -1)
            cv2.putText(frame, "SYNTHETIC TEST", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
            writer.write(frame)
    finally:
        writer.release()
    print(f"Created synthetic MOV: {args.output}")


if __name__ == "__main__":
    main()
