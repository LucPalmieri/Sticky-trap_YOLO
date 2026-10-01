#!/usr/bin/env python3
"""
Mark the classifier's positive detections on the flatbug overview and emit a
bench-location table, so you can find each target insect on the physical trap
for DNA extraction.

Inputs it joins:
  - classifications.csv  (from classify_crops.py) -> which CROPNUMBERs are positive
  - metadata_*.json      (from flatbug)           -> each detection's contour + box
  - overview_*.jpg       (from flatbug)           -> image to draw on
The crop's pixel dimensions were verified to map 1:1 to a metadata box - see
resolve_crop_index.py, which must be run once (after training a new
classifier, upgrading flatbug, or switching source dataset) to confirm
CROPNUMBER == metadata index for your data. This script trusts that mapping
at draw time; it does not re-verify it per run.

Outputs:
  - overview_positives.jpg : positives outlined thick bright-green + labelled
  - positives_bench.csv    : cropnumber, box, centre px, % across/down, (cm if --card given)
  - context/ (optional)    : high-res cutout around each positive from the overview

A trap with zero positives for --target is a normal, expected outcome (most
traps have none) - this exits 0 with a message, not an error.

Example:
  python highlight_positives.py \
    --csv classifications.csv \
    --json metadata_Trap_big1_UUID_2424e404-ab5f-49e5-8952-826559198af4.json \
    --overview overview_Trap_big1_UUID_2424e404-ab5f-49e5-8952-826559198af4.jpg \
    --outdir . --card-w 20 --card-h 25 --context
"""
import argparse
import csv
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
GREEN = (0, 255, 0)


def load_positive_cropnumbers(csv_path, target, flag):
    """CROPNUMBERs whose row is a positive (flag==TARGET, or label==target)."""
    out = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = (row.get("flag") or "").strip()
            lab = (row.get("label") or "").strip()
            if f == flag or (flag == "" and lab == target):
                m = re.search(r"CROPNUMBER_(\d+)_", row["image_file"])
                if m:
                    out.append((int(m.group(1)), lab, float(row["confidence"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--overview", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--target", default="Pentastiridius")
    ap.add_argument("--flag", default="TARGET", help="CSV flag marking a positive")
    ap.add_argument("--card-w", type=float, default=None, help="physical card width (cm)")
    ap.add_argument("--card-h", type=float, default=None, help="physical card height (cm)")
    ap.add_argument("--width", type=int, default=10, help="outline thickness (px)")
    ap.add_argument("--context", action="store_true", help="also cut a context view per positive")
    ap.add_argument("--pad", type=int, default=400, help="context margin around box (px)")
    args = ap.parse_args()

    d = json.load(open(os.path.expanduser(args.json)))
    boxes, contours = d["boxes"], d["contours"]
    W, H = d["image_width"], d["image_height"]

    positives = load_positive_cropnumbers(os.path.expanduser(args.csv), args.target, args.flag)
    if not positives:
        print(f"no '{args.flag}' rows for target '{args.target}' in {args.csv} "
              f"- nothing to highlight (this is a normal outcome, not an error)")
        return

    over = Image.open(os.path.expanduser(args.overview)).convert("RGB")
    if over.size != (W, H):
        print(f"[note] overview {over.size} != scan ({W}x{H}); scaling contours by that ratio")
    sx, sy = over.size[0] / W, over.size[1] / H
    draw = ImageDraw.Draw(over)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 120)
    except Exception:
        font = ImageFont.load_default()

    os.makedirs(os.path.expanduser(args.outdir), exist_ok=True)
    if args.context:
        os.makedirs(os.path.join(os.path.expanduser(args.outdir), "context"), exist_ok=True)

    rows = []
    for cn, lab, conf in sorted(positives):
        if cn >= len(boxes):
            print(f"[warn] crop {cn} beyond metadata range - skipped")
            continue
        b = boxes[cn]
        # trusts CROPNUMBER == metadata index (validated once via
        # resolve_crop_index.py - a silent mismatch here would mislabel a DNA
        # target, so re-run that check after any flatbug/classifier upgrade)
        poly = contours[cn]
        # flatbug stores each contour as two parallel arrays: [[x0,x1,...],[y0,y1,...]]
        xs, ys = poly[0], poly[1]
        spts = [(x * sx, y * sy) for x, y in zip(xs, ys)]
        if len(spts) >= 2:
            draw.line(spts + [spts[0]], fill=GREEN, width=args.width)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        draw.text((cx * sx + 15, cy * sy - 140), f"{cn}", fill=GREEN, font=font)

        fx, fy = cx / W, cy / H
        rec = {"cropnumber": cn, "label": lab, "confidence": round(conf, 4),
               "box_x0": b[0], "box_y0": b[1], "box_x1": b[2], "box_y1": b[3],
               "centre_x_px": int(cx), "centre_y_px": int(cy),
               "pct_across": round(100 * fx, 1), "pct_down": round(100 * fy, 1)}
        if args.card_w and args.card_h:
            rec["cm_right"] = round(fx * args.card_w, 2)
            rec["cm_down"] = round(fy * args.card_h, 2)
        rows.append(rec)

        if args.context:
            x0 = max(0, b[0] - args.pad); y0 = max(0, b[1] - args.pad)
            x1 = min(W, b[2] + args.pad); y1 = min(H, b[3] + args.pad)
            cut = Image.open(os.path.expanduser(args.overview)).convert("RGB").crop((x0, y0, x1, y1))
            cut.save(os.path.join(os.path.expanduser(args.outdir), "context", f"positive_{cn}.jpg"))

    if not rows:
        print("no positives survived metadata-range validation - nothing written")
        return

    out_img = os.path.join(os.path.expanduser(args.outdir), "overview_positives.jpg")
    over.save(out_img, quality=90)
    out_csv = os.path.join(os.path.expanduser(args.outdir), "positives_bench.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"marked {len(rows)} positives -> {out_img}")
    print(f"bench table -> {out_csv}\n")
    hdr = "crop  conf    centre(px)        %across %down"
    if args.card_w and args.card_h:
        hdr += "   cm_right cm_down"
    print(hdr)
    for r in rows:
        line = (f"{r['cropnumber']:>4}  {r['confidence']:.3f}  "
                f"({r['centre_x_px']:>6},{r['centre_y_px']:>6})  "
                f"{r['pct_across']:>6}  {r['pct_down']:>6}")
        if args.card_w and args.card_h:
            line += f"   {r['cm_right']:>7}  {r['cm_down']:>6}"
        print(line)


if __name__ == "__main__":
    main()
