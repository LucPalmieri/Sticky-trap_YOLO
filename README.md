# Sticky-trap insect screening pipeline

> 🚧 **Under construction** - in active implementation and field validation.

Automated screening of high-resolution sticky-trap scans for
*Pentastiridius leporinus*, the planthopper vector of Syndrome Basses
Richesses (SBR) in sugar beet. The pipeline finds every insect on a scanned
trap, classifies each one, and produces a map of candidate vectors, so they
can be located on the physical card and confirmed by eye and by DNA
barcoding.

## How it works

```
 trap scan (JPEG, ~3200 dpi)
        │
        ▼
 1. flatbug ─────────────── detects + segments every arthropod, writes one crop
        │                   per insect (species-agnostic, runs once per scan)
        ▼
 2. classify_crops.py ───── YOLO11s-cls classifier, 15 classes, trained on the
        │                   SBR Cixiidae dataset; flags the target as a candidate
        ▼
 3. copy_positives.py ───── gathers flagged crops into positive/ for review
        │
        ▼
 4. highlight_positives.py  outlines candidates on the trap overview + writes a
                            bench table (pixel / % / cm position on the card)
```

`scripts/run_pipeline.sh` chains all four steps. It runs locally, or as a
SLURM array with one task per scan, so a scan that fails or runs out of
memory doesn't stop the others. Classifiers are pluggable: each target
species is a folder in `models/` (see [`models/README.md`](models/README.md)),
and a second species classifier (*Scaphoideus titanus*) is planned.

## Status and limitations

- On the source dataset's held-out trap cards the classifier separates
  *Pentastiridius* well. On our own field traps (2026 field trial) it has
  so far flagged many specimens that look like **Diptera** on visual
  inspection, some with high confidence. The leading hypothesis is a
  trap-product/imaging difference from the training data. COI barcoding of
  the flagged specimens is under way.
- Treat every `TARGET` flag as a **candidate for confirmation**, not an
  identification.
- `build_card_level_split.py` exists because the dataset's original split
  puts crops from the same trap card in train, val and test, which inflates
  held-out scores. Train on the card-level split.

## Repository layout

```
scripts/       pipeline + training scripts
models/        one folder per target species (config only; weights not included)
environment/   requirements (CPU) and GPU/HPC conda setup
sample_data/
  pentastiridius/trap01-06/   real field crops + expected classifier output
  scans_demo/                 six 3000x3000 px windows of real trap scans
```

## Installation

Tested on Linux (Ubuntu / WSL2), Python 3.12, CPU only. A GPU is
recommended for real scans (see `environment/setup_flatbug_env.sh`).

```bash
git clone https://github.com/LucPalmieri/Sticky-trap_YOLO.git && cd Sticky-trap_YOLO
python3 -m venv .venv && source .venv/bin/activate

# CPU build of PyTorch (skip for GPU and install the CUDA build instead)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# flatbug (source install; it pins ultralytics<=8.3.124)
git clone https://github.com/darsa-group/flat-bug.git
pip install -e flat-bug

# classifier + post-processing deps
pip install -r environment/requirements-cpu.txt
```

If `fb_predict` fails with *"Failed to find C compiler"*, install `gcc` or
`export TORCHINDUCTOR_DISABLE=1 TORCHDYNAMO_DISABLE=1`.

## What to download

No model weights or training data are redistributed here. You need:

| What | Where it goes | Source |
|---|---|---|
| flatbug detector weights `flat_bug_M.pt` | `models/_flatbug/flat_bug_M.pt` | [flatbug](https://github.com/darsa-group/flat-bug) model repository: <https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/flat_bug_M.pt> |
| SBR Cixiidae and Associated Insects Dataset (N. Okole, CC BY 4.0) | anywhere, e.g. `data/SBR_Cixiidae_Dataset/` | [Kaggle](https://www.kaggle.com/datasets/nathanokole/high-resolution-image-of-insects-on-sticky-traps) - see also the [paper](https://link.springer.com/article/10.1007/s41348-026-01275-6) |
| classifier weights `best.pt` | `models/pentastiridius/best.pt` | train it yourself from the dataset (below) |

```bash
mkdir -p models/_flatbug
wget -O models/_flatbug/flat_bug_M.pt \
  https://anon.erda.au.dk/share_redirect/Bb0CR1FHG6/models/flat_bug_M.pt
```

## Train the classifier

```bash
# 1. regroup the dataset so each physical trap card sits in one split only
python scripts/build_card_level_split.py \
    --src data/SBR_Cixiidae_Dataset --dst data/SBR_Cixiidae_cardsplit

# 2. train YOLO11s-cls (imgsz 320, 120 epochs, full rotation augmentation)
python scripts/train_sbr_classifier.py --data data/SBR_Cixiidae_cardsplit --device 0

# 3. put the weights where the pipeline looks for them
cp runs/classify/sbr_cixiidae_cls/weights/best.pt models/pentastiridius/best.pt
```

Training needs a GPU in practice (`--device cpu` works, slowly). Check the
per-class confusion matrix, not top-1 accuracy: the classes are heavily
imbalanced.

## Run it

**1. Crop-level check (seconds, no flatbug needed).** Classify the sample
crops and compare with the reference output:

```bash
python scripts/classify_crops.py -w models/pentastiridius/best.pt \
    -i sample_data/pentastiridius/trap01/crops -o out.csv --device cpu
diff --strip-trailing-cr out.csv sample_data/pentastiridius/trap01/expected_classifications.csv
```

The reference CSVs were made with our trained weights. With weights you
trained yourself, expect similar but not identical numbers.

**2. Full pipeline on the demo scans (a few minutes per image on CPU).**

```bash
DEVICE=cpu bash scripts/run_pipeline.sh sample_data/scans_demo out/
```

Each demo window is centred on a candidate from the field trial, so a
candidate is expected near the middle (see
[`sample_data/scans_demo/README.md`](sample_data/scans_demo/README.md)).

**3. Your own scans.**

```bash
# flatbug reads JPEG/PNG only - convert TIFF scans first
bash scripts/tif_to_jpg.sh scans_tif/ scans_jpg/

# locally
DEVICE=0 bash scripts/run_pipeline.sh scans_jpg/ results/

# or on SLURM: one array task per scan (edit the #SBATCH header for your site)
mkdir -p logs
N=$(ls scans_jpg/*.jpg | wc -l)
sbatch --array=0-$((N-1))%4 scripts/run_pipeline.sh scans_jpg/ results/
```

Options are environment variables: `DEVICE` (GPU index or `cpu`),
`CONDA_ENV` or `VENV` (environment to activate), `CARD_W` / `CARD_H`
(card size in cm, adds cm positions to the bench table), `SCALE` (flatbug
downscale for very large scans), `MODELS_DIR`, `FB_WEIGHTS`.

### Outputs (per scan)

```
results/<scan>/
  crops/                      one image per detected insect (flatbug)
  metadata_*.json             contours + boxes (flatbug)
  overview_*.jpg              scan with all detections (flatbug)
  pentastiridius/
    classifications.csv       label, confidence, top-3, flag for every crop
    positive/                 crops flagged TARGET
    overview_positives.jpg    candidates outlined + numbered on the scan
    positives_bench.csv       candidate positions for finding them on the card
    context/                  zoomed cut-out around each candidate
```

`scripts/resolve_crop_index.py` is a one-off check that flatbug crop numbers
line up with the metadata index. Re-run it after upgrading flatbug.

## Acknowledgements and citation

- **flatbug** - darsa-group, MIT licence. Please cite the flatbug paper:
  [doi:10.1111/2041-210x.70249](https://doi.org/10.1111/2041-210x.70249).
- **SBR Cixiidae and Associated Insects Dataset** - Nathan Okole (IfZ
  Göttingen), CC BY 4.0.
- **Ultralytics YOLO** - AGPL-3.0.

See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

## License

Code in this repository: MIT (see [`LICENSE`](LICENSE)). Third-party
components keep their own licences.
