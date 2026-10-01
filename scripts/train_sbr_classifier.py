#!/usr/bin/env python3
"""
Train a YOLO11 IMAGE CLASSIFIER on Okole's SBR_Cixiidae_Dataset so the
single-insect crops that flatbug produces get sensible taxon names.

Why classification (not detection): flatbug already localises and crops each
insect. The remaining job is "one crop in -> one taxon out", which is
classification. Feeding tight single-insect crops to a *detector* is what
produced the "No Detections Found" wall in the IP102 run.

The dataset already ships as train/ val/ test/ with one subfolder per class,
so there is nothing to crop or parse - point --data at the dataset root.

Tested against ultralytics 8.4.98.

Example:
    python train_sbr_classifier.py --data ~/datasets/SBR_Cixiidae_Dataset --device 0
"""
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def class_counts(split_dir: Path) -> Counter:
    c = Counter()
    if not split_dir.is_dir():
        return c
    for d in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        c[d.name] = sum(1 for f in d.glob("*") if f.suffix.lower() in IMG_EXT)
    return c


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="SBR_Cixiidae_Dataset root (contains train/ val/ test/)")
    ap.add_argument("--model", default="yolo11s-cls.pt",
                    help="Pretrained classifier. Bump to yolo11m-cls.pt if the "
                         "cixiid genera stay confused at higher imgsz.")
    ap.add_argument("--imgsz", type=int, default=320,
                    help="320+ helps separate the look-alike cixiid genera "
                         "(Pentastiridius/Hyalesthes/Reptalus). Try 384 if that "
                         "3x3 block in the confusion matrix stays muddy. The "
                         "ImageNet default of 224 tends to blur fine venation.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4,
                    help="dataloader workers. Each is a process holding decoded, "
                         "prefetched crops - the main host-RAM cost. Lower to 2 if "
                         "the Linux OOM killer takes the run (common on WSL's capped "
                         "RAM); raise if you have memory headroom and want speed.")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="sbr_cixiidae_cls")
    args = ap.parse_args()

    data = Path(args.data).expanduser().resolve()

    # --- Report class balance BEFORE training -------------------------------
    counts = class_counts(data / "train")
    if not counts:
        raise SystemExit(f"No class folders under {data / 'train'} - check --data")
    total = sum(counts.values())
    if total == 0:
        raise SystemExit(
            f"Found {len(counts)} class folders but 0 images with a recognised "
            f"extension under {data / 'train'}. Recognised: {sorted(IMG_EXT)}")
    peak = max(counts.values())
    print(f"\nTraining crops per class (total {total}):")
    for cls, n in counts.most_common():
        bar = "#" * int(40 * n / peak)
        print(f"  {n:6d} {n / total:6.1%}  {cls:16s} {bar}")
    print(f"\nImbalance ratio (max/min): {peak / max(1, min(counts.values())):.0f}x")
    rare = [c for c, n in counts.items() if n < 50]
    if rare:
        print(f"[warn] <50 crops (expect weak recall): {rare}")
        print("       -> consider oversampling these folders first, and judge")
        print("          them on per-class recall, NOT global top-1 accuracy.\n")

    # --- Train --------------------------------------------------------------
    model = YOLO(args.model)
    model.train(
        data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        workers=args.workers, device=args.device, patience=25, name=args.name,
        # insects land on cards at every orientation -> full rotational aug
        fliplr=0.5, flipud=0.5, degrees=180.0,
    )

    # --- Evaluate on the held-out TEST split --------------------------------
    # Global top-1 is dominated by the common classes and will look good even if
    # Pentastiridius recall is poor. Read the confusion matrix, not the headline.
    m = model.val(data=str(data), split="test", imgsz=args.imgsz, device=args.device)
    print(f"\nTEST top-1: {float(m.top1):.3f}   top-5: {float(m.top5):.3f}")
    print("Open runs/classify/*/confusion_matrix.png and check, in order:")
    print("  1. Pentastiridius recall (its row) - the class you actually care about")
    print("  2. the Pentastiridius / Hyalesthes / Reptalus 3x3 block - the hard call")
    print("  3. how much real insect signal leaks into Artefact / Other insects")


if __name__ == "__main__":
    main()
