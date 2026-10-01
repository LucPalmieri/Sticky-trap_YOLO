#!/usr/bin/env python3
"""
Copy the crops flagged TARGET (positive for --target) into a 'positive/' folder
next to the crops, so they can be eyeballed together.

Example:
    python copy_positives.py \
        --csv   big_output/Trap_big1/classifications.csv \
        --crops big_output/Trap_big1/crops \
        --dest  big_output/Trap_big1/positive
"""
import argparse
import csv
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="classifications.csv from classify_crops.py")
    ap.add_argument("--crops", required=True, help="folder holding the crop images")
    ap.add_argument("--dest", required=True, help="folder to copy positives into")
    ap.add_argument("--flag", default="TARGET", help="flag value to select (default TARGET)")
    args = ap.parse_args()

    crops = os.path.expanduser(args.crops)
    dest = os.path.expanduser(args.dest)
    os.makedirs(dest, exist_ok=True)

    copied, missing = 0, []
    # newline="" + default reader handles the CRLF the CSV was written with
    with open(os.path.expanduser(args.csv), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("flag") or "").strip() != args.flag:
                continue
            name = row["image_file"].strip()
            src = os.path.join(crops, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, name))
                copied += 1
                print(f"{row['label'].strip():15s} {float(row['confidence']):.4f}  {name}")
            else:
                missing.append(name)

    print(f"\ncopied {copied} '{args.flag}' crops -> {dest}")
    if missing:
        print(f"[warn] {len(missing)} listed crops not found in {crops}, e.g. {missing[0]}")


if __name__ == "__main__":
    main()
