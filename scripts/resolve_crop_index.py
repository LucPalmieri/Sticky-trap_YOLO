#!/usr/bin/env python3
"""
Resolve each positive crop to its detection in the flatbug metadata JSON by
matching the crop's pixel dimensions (which equal its bounding-box w x h) to a
box in the metadata. The crop's CROPNUMBER is NOT the array index, so we can't
index directly - we match on geometry and cross-check position.

Prints, per positive crop: the metadata index it maps to, the box, and whether
the match is unambiguous. Run this BEFORE drawing anything - if any crop has 0
or >1 candidate matches, the naive mapping is unsafe and we handle it before
touching the overview.

One-time check, not a pipeline stage: run it once against a fresh flatbug
version, a new classifier, or a new source dataset to confirm CROPNUMBER ==
metadata index still holds before trusting highlight_positives.py's output.
It is deliberately not wired into run_pipeline.sh.
"""
import argparse
import glob
import json
import os
import re

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="metadata_*.json from flatbug")
    ap.add_argument("--crops", required=True, help="folder of positive crops to resolve")
    ap.add_argument("--tol", type=int, default=2, help="pixel tolerance on w/h match")
    args = ap.parse_args()

    d = json.load(open(os.path.expanduser(args.json)))
    boxes = d["boxes"]
    print(f"metadata: {len(boxes)} detections, scan {d['image_width']}x{d['image_height']}\n")

    crops = sorted(glob.glob(os.path.join(os.path.expanduser(args.crops), "*CROPNUMBER_*")))
    if not crops:
        raise SystemExit(f"no crops found in {args.crops}")

    for path in crops:
        cn = re.search(r"CROPNUMBER_(\d+)_", os.path.basename(path)).group(1)
        w, h = Image.open(path).size
        # candidate metadata indices whose box dimensions match this crop
        cands = [i for i, b in enumerate(boxes)
                 if abs((b[2] - b[0]) - w) <= args.tol and abs((b[3] - b[1]) - h) <= args.tol]
        flag = "OK" if len(cands) == 1 else ("NONE" if not cands else f"AMBIGUOUS({len(cands)})")
        detail = ""
        if len(cands) == 1:
            b = boxes[cands[0]]
            detail = f" box={b}"
        elif len(cands) > 1:
            detail = f" indices={cands}"
        print(f"crop {cn:>4}  ({w}x{h})  -> metadata index {cands if cands else '-'}  [{flag}]{detail}")

    print("\nAll 'OK' -> geometry mapping is safe. Any NONE/AMBIGUOUS -> resolve it\n"
          "with box position (centroid) before trusting highlight_positives.py.")


if __name__ == "__main__":
    main()
