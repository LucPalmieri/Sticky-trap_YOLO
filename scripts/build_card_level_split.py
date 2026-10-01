#!/usr/bin/env python3
"""
Rebuild the SBR dataset with a CARD-LEVEL train/val/test split.

Okole's shipped split scatters every physical trap card across all three splits
(verified on this dataset: 278/278 cards span train+val+test), so its held-out
numbers are optimistic - the model can lean on per-card background/lighting/scan
cues instead of morphology. This regroups crops so each card lands wholly in ONE
split, while keeping the class mix balanced across splits via StratifiedGroupKFold
(so the rare cixiid vectors don't vanish from val/test).

The source tree is left untouched: the new tree is built from symlinks by default
(no copy, no duplication). Point train_sbr_classifier.py at --dst afterwards.

Needs: pandas, scikit-learn.

Example:
    python build_card_level_split.py --src ~/SBR_Cixiidae_Dataset \
                                     --dst ~/SBR_Cixiidae_cardsplit
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# maps the original top-level split folder (as written in metadata 'path') to the
# lowercase folder names you copied them into
SPLIT_MAP = {"Train": "train", "Validation": "val", "Test": "test"}


def current_path(src: Path, rel: str) -> Path:
    """Reconstruct where a crop physically lives now from its metadata path."""
    parts = rel.replace("\\", "/").split("/")
    parts[0] = SPLIT_MAP.get(parts[0], parts[0].lower())
    return src.joinpath(*parts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="current dataset root (has metadata.csv)")
    ap.add_argument("--dst", required=True, help="new card-split tree to create")
    ap.add_argument("--n-splits", type=int, default=5,
                    help="StratifiedGroupKFold folds; fold0->test, fold1->val, "
                         "rest->train (5 => ~60/20/20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (needed if the tree "
                         "will cross a filesystem that won't follow symlinks)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    meta = src / "metadata.csv"
    if not meta.exists():
        sys.exit(f"metadata.csv not found in {src}")
    if dst.exists():
        sys.exit(f"{dst} already exists - remove it or pick another --dst")

    df = pd.read_csv(meta)
    df["trap"] = df["path"].str.extract(r"(TR\d+)", expand=False)
    if df["trap"].isna().any():
        sys.exit(f"{df['trap'].isna().sum()} rows have no TR id - can't define cards")
    df["card"] = (df["location"].astype(str) + "|" + df["date"].astype(str)
                  + "|" + df["trap"].astype(str))

    # resolve + verify every crop is actually on disk before touching anything
    df["cur"] = df["path"].map(lambda r: current_path(src, r))
    missing = df["cur"].map(lambda p: not p.exists())
    if missing.any():
        sys.exit(f"{int(missing.sum())} files in metadata not found on disk, e.g.\n"
                 f"  {df.loc[missing, 'cur'].iloc[0]}")

    # basenames are NOT globally unique in this dataset - the same crop name
    # recurs across the original Train/Validation/Test folders. Prefix every
    # destination with its original split folder so nothing collides or is lost.
    n_dup = int(df["cur"].map(lambda p: p.name).duplicated(keep=False).sum())
    if n_dup:
        print(f"[note] {n_dup} crops share a basename across the original splits; "
              f"prefixing destination names with the original split to keep them "
              f"distinct (no files dropped).")

    # --- card-level, class-stratified fold assignment -----------------------
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True,
                                random_state=args.seed)
    fold = pd.Series(-1, index=df.index)
    for f, (_, idx) in enumerate(sgkf.split(df, df["label"], df["card"])):
        fold.iloc[idx] = f
    df["split"] = fold.map({0: "test", 1: "val"}).fillna("train")

    # --- build the tree -----------------------------------------------------
    n = 0
    for row in df.itertuples(index=False):
        out = dst / row.split / row.label
        out.mkdir(parents=True, exist_ok=True)
        link = out / f"{row.cur.parts[-3]}__{row.cur.name}"  # orig-split prefix
        if args.copy:
            shutil.copy2(row.cur, link)
        else:
            os.symlink(row.cur, link)  # absolute target -> resolves from anywhere
        n += 1
    df.drop(columns=["cur"]).to_csv(dst / "metadata_cardsplit.csv", index=False)

    # --- verify the leak is actually gone, then report ----------------------
    spans = df.groupby("card")["split"].nunique()
    if not (spans == 1).all():
        sys.exit("FATAL: some cards still span splits - do not trust this tree")
    print(f"built {n} {'copies' if args.copy else 'symlinks'} under {dst}")
    print(f"cards: {df['card'].nunique()}   cards spanning >1 split: "
          f"{int((spans > 1).sum())}  [clean]\n")

    counts = (df.pivot_table(index="label", columns="split", values="path",
                             aggfunc="count", fill_value=0)
                .reindex(columns=["train", "val", "test"], fill_value=0))
    counts["cards"] = df.groupby("label")["card"].nunique()
    print(counts.sort_values("train", ascending=False).to_string())

    thin = counts[(counts[["val", "test"]] == 0).any(axis=1)]
    if len(thin):
        print("\n[warn] classes missing from val and/or test (too few cards to "
              "spread):", thin.index.tolist())
        print("       That's a dataset limitation - those classes can't be")
        print("       honestly evaluated no matter how you split. Note it; don't")
        print("       paper over it by reverting to a crop-level split.")


if __name__ == "__main__":
    main()
