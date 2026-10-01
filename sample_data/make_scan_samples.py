#!/usr/bin/env python3
"""
One-time data-prep helper - NOT part of the deployed pipeline.

Cuts a small, native-resolution window out of each real full-scan trap image
under scan_samples/, centred on that trap's highest-confidence Pentastiridius
candidate (from the real positives_bench.csv produced during the field
trial). Used to build sample_data/scans_demo/ from source material that is
otherwise too large to commit (the originals are 100MB+ each).

Why a crop and not a downscale: shrinking a whole ~30000x25000 scan risks
shrinking a millimetric planthopper below flatbug's effective detection
size. Cropping at full native resolution keeps every pixel the detector
would have seen; it just narrows the field of view.

Not meant to be re-run generically - paths are specific to this repo's
scan_samples/ layout. Kept for provenance/reproducibility, not reuse.

Usage:
    python make_scan_samples.py
"""
import csv
import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "scan_samples", "Pentastiridius")
OUT = os.path.join(BASE, "sample_data", "scans_demo")
WIN = 3000  # crop window side, px - native resolution, no downscaling
TRAPS = ["trap01", "trap02", "trap03", "trap04", "trap05", "trap06"]


def best_positive(bench_path):
    rows = list(csv.DictReader(open(bench_path)))
    return max(rows, key=lambda r: float(r["confidence"]))


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for t in TRAPS:
        img_path = os.path.join(RAW, f"{t}_sample.jpg")
        bench_path = os.path.join(RAW, "crops_demo", t, "positives_bench.csv")
        im = Image.open(img_path)
        W, H = im.size
        r = best_positive(bench_path)
        cx, cy = int(r["centre_x_px"]), int(r["centre_y_px"])
        half = WIN // 2
        x0 = max(0, min(cx - half, W - WIN))
        y0 = max(0, min(cy - half, H - WIN))
        crop = im.crop((x0, y0, x0 + WIN, y0 + WIN))
        out_path = os.path.join(OUT, f"{t}.jpg")
        crop.save(out_path, "JPEG", quality=92)
        size_mb = os.path.getsize(out_path) / 1e6
        manifest.append({
            "file": f"{t}.jpg", "source_scan": f"{t}_sample.jpg",
            "anchor_cropnumber": r["cropnumber"], "anchor_confidence": r["confidence"],
            "anchor_pos_in_crop_px": f"{cx - x0},{cy - y0}",
            "crop_origin_in_source_px": f"{x0},{y0}", "size_mb": round(size_mb, 2),
        })
        print(f"{t}: anchor cn={r['cropnumber']} conf={r['confidence']} "
              f"-> {out_path} ({size_mb:.2f}MB)")

    manifest_path = os.path.join(OUT, "manifest.csv")
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"\nmanifest -> {manifest_path}")


if __name__ == "__main__":
    main()
