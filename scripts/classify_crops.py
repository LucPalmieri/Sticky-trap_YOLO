#!/usr/bin/env python3
"""
Classify every flatbug crop in a folder with a trained YOLO11-cls model and
write one CSV row per crop: top-1 label, confidence, top-3, and a flag.

  flag = TARGET  -> top-1 is --target with confidence >= --min-conf
  flag = review  -> top-1 confidence < --min-conf (label set to 'uncertain')

A TARGET flag is a candidate for visual / molecular confirmation, not an ID.

Example:
    python classify_crops.py -w models/pentastiridius/best.pt \
        -i out/trap01/crops -o out/trap01/classifications.csv --device cpu
"""
import argparse
import csv
import glob
import os

from ultralytics import YOLO

EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-w", "--weights", required=True, help="trained best.pt")
    ap.add_argument("-i", "--crops", required=True, help="directory of flatbug crops")
    ap.add_argument("-o", "--out", required=True, help="output CSV path")
    ap.add_argument("--target", default="Pentastiridius",
                    help="class name to flag as a hit")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="below this top-1 confidence -> 'uncertain'")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    crops = []
    for e in EXTS:
        crops.extend(glob.glob(os.path.join(os.path.expanduser(args.crops), e)))
    crops = sorted(set(crops))
    if not crops:
        raise SystemExit(f"No crops found in {args.crops}")

    model = YOLO(os.path.expanduser(args.weights))
    known = set(model.names.values())
    if args.target not in known:
        print(f"[warn] --target '{args.target}' is not a class of this model.")
        print(f"       classes: {sorted(known)}")

    n_target = 0
    out = os.path.expanduser(args.out)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_file", "label", "confidence", "top3", "flag"])
        for path in crops:
            r = model(path, verbose=False, device=args.device)[0]
            probs = r.probs
            conf = float(probs.top1conf)
            order = probs.data.argsort(descending=True)[:3].tolist()
            top3 = "; ".join(f"{r.names[i]}:{float(probs.data[i]):.2f}" for i in order)

            if conf < args.min_conf:
                label, flag = "uncertain", "review"
            else:
                label = r.names[probs.top1]
                flag = ""
                if label == args.target:
                    flag = "TARGET"
                    n_target += 1
            w.writerow([os.path.basename(path), label, f"{conf:.4f}", top3, flag])

    print(f"{len(crops)} crops -> {out}")
    print(f"{args.target} hits above {args.min_conf:g}: {n_target}")


if __name__ == "__main__":
    main()
