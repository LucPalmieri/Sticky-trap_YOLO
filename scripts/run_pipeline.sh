#!/bin/bash
# =============================================================================
# Sticky-trap vision pipeline - ONE SLURM array task per trap image.
#
# Per-image isolation is deliberate: if one huge scan runs out of memory, only
# that task dies - every other trap still completes. Each task runs:
#
#   flatbug (detect+crop, ONCE per image - detection is species-agnostic)
#     -> for each configured pest model under models/:
#          classify_crops -> copy_positives -> highlight_positives
#
# Adding a new target species = drop a models/<name>/{weights,config.sh}
# folder in place (see models/README.md). Nothing in this script changes.
#
# SUBMIT (SLURM):
#   N=$(find scans -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \
#        -o -iname '*.png' \) | wc -l)
#   mkdir -p logs
#   sbatch --array=0-$((N-1))%4 scripts/run_pipeline.sh <scans_dir> <out_dir>   # from repo root
# The %4 caps concurrent tasks at 4 (tune to your GPU/memory budget).
#
# RUN LOCALLY (no SLURM): processes every image in <scans_dir> in turn
#   DEVICE=cpu bash scripts/run_pipeline.sh <scans_dir> <out_dir>
#
# Input must be JPEG/PNG - flatbug cannot read TIFF (use tif_to_jpg.sh first).
# =============================================================================
# Set these for your cluster (account / partition are site-specific):
##SBATCH --account=<your_account>
##SBATCH --partition=<gpu_partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=sticky_trap_pipeline
#SBATCH --output=logs/pipeline_%A_%a.out
#SBATCH --error=logs/pipeline_%A_%a.err

set -uo pipefail

# ---- paths & options (override any of these with env vars) -----------------
# Under sbatch, BASH_SOURCE points at SLURM's spooled copy of this script, so
# PROJ defaults to the submit directory: submit from the repo root, or export
# PROJ=/path/to/repo.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    PROJ=${PROJ:-$SLURM_SUBMIT_DIR}
else
    PROJ=${PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
fi
SCRIPTS=${SCRIPTS:-$PROJ/scripts}
SCANS_DIR=${1:?usage: run_pipeline.sh <scans_dir> <out_dir>}
OUT_DIR=${2:?usage: run_pipeline.sh <scans_dir> <out_dir>}
FB_WEIGHTS=${FB_WEIGHTS:-$PROJ/models/_flatbug/flat_bug_M.pt}  # detector (species-agnostic)
MODELS_DIR=${MODELS_DIR:-$PROJ/models}                         # one subfolder per pest
DEVICE=${DEVICE:-0}                   # GPU index, or "cpu"
CARD_W=${CARD_W:-}                    # optional: physical card width  (cm)
CARD_H=${CARD_H:-}                    # optional: physical card height (cm)
SCALE=${SCALE:-}                      # optional: flatbug downscale, e.g. 0.5 for giant scans

# ---- environment (optional; conda hook is required in batch mode) ---------
# CONDA_ENV=<name or path>  -> conda activate it
# VENV=<path>               -> source its bin/activate
# ENVFILE=<file>            -> sourced afterwards (site-specific modules, cache dirs...)
# none set                  -> use whatever environment is already active
if [ -n "${CONDA_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
    conda activate "$CONDA_ENV"
elif [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
# shellcheck disable=SC1090
[ -n "${ENVFILE:-}" ] && source "$ENVFILE"
mkdir -p "$OUT_DIR"

if [ ! -f "$FB_WEIGHTS" ]; then
    echo "[error] flatbug weights not found: $FB_WEIGHTS (see README)"; exit 1
fi
PEST_DIRS=$(find "$MODELS_DIR" -mindepth 2 -maxdepth 2 -name 'config.sh' 2>/dev/null | sort)
if [ -z "$PEST_DIRS" ]; then
    echo "[error] no pest models with a config.sh found under $MODELS_DIR"; exit 1
fi

process_one() {
    local IMG=$1
    local NAME; NAME=$(basename "$IMG"); NAME="${NAME%.*}"
    echo "=== $IMG  (name=$NAME) ==="

    # ---- 1. flatbug: detect + crop this ONE image, shared by every pest below
    local SCALE_ARG=""; [ -n "$SCALE" ] && SCALE_ARG="-s $SCALE"
    # shellcheck disable=SC2086
    fb_predict -i "$IMG" -o "$OUT_DIR" -w "$FB_WEIGHTS" --device "$DEVICE" $SCALE_ARG
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[error] flatbug exited $rc for $NAME - stopping this image"; return 1
    fi
    local CROPS="$OUT_DIR/$NAME/crops"
    if [ ! -d "$CROPS" ] || [ -z "$(ls -A "$CROPS" 2>/dev/null)" ]; then
        echo "flatbug produced no crops for $NAME - stopping this image"; return 1
    fi
    local META OVER CARD_ARGS=""
    META=$(ls "$OUT_DIR/$NAME"/metadata_*.json 2>/dev/null | head -1)
    OVER=$(ls "$OUT_DIR/$NAME"/overview_*.jpg 2>/dev/null | head -1)
    [ -n "$CARD_W" ] && [ -n "$CARD_H" ] && CARD_ARGS="--card-w $CARD_W --card-h $CARD_H"

    # ---- 2-4. classify + extract positives + localise, once per pest model
    local STATUS=0 CONFIG PEST_DIR PEST CLS_WEIGHTS PEST_OUT
    for CONFIG in $PEST_DIRS; do
        PEST_DIR=$(dirname "$CONFIG")
        PEST=$(basename "$PEST_DIR")

        # reset to documented defaults before sourcing, so one pest's config.sh
        # can't leak values into the next iteration
        TARGET=""; MIN_CONF="0.5"; WEIGHTS="best.pt"
        # shellcheck disable=SC1090
        source "$CONFIG"
        if [ -z "$TARGET" ]; then
            echo "[warn] $CONFIG did not set TARGET - skipping $PEST"; continue
        fi
        CLS_WEIGHTS="$PEST_DIR/$WEIGHTS"
        if [ ! -f "$CLS_WEIGHTS" ]; then
            echo "[warn] $CLS_WEIGHTS missing - skipping $PEST"; continue
        fi

        PEST_OUT="$OUT_DIR/$NAME/$PEST"
        mkdir -p "$PEST_OUT"
        echo "--- $NAME / $PEST (target=$TARGET, min-conf=$MIN_CONF) ---"

        python "$SCRIPTS/classify_crops.py" \
            -w "$CLS_WEIGHTS" -i "$CROPS" \
            -o "$PEST_OUT/classifications.csv" \
            --target "$TARGET" --min-conf "$MIN_CONF" --device "$DEVICE"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[error] classify_crops.py failed for $PEST (exit $rc)"; STATUS=1; continue
        fi

        python "$SCRIPTS/copy_positives.py" \
            --csv "$PEST_OUT/classifications.csv" \
            --crops "$CROPS" \
            --dest "$PEST_OUT/positive"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[error] copy_positives.py failed for $PEST (exit $rc)"; STATUS=1; continue
        fi

        if [ -n "$META" ] && [ -n "$OVER" ]; then
            # shellcheck disable=SC2086
            python "$SCRIPTS/highlight_positives.py" \
                --csv "$PEST_OUT/classifications.csv" \
                --json "$META" --overview "$OVER" --target "$TARGET" \
                --outdir "$PEST_OUT" --context $CARD_ARGS
            rc=$?
            if [ $rc -ne 0 ]; then
                echo "[error] highlight_positives.py failed for $PEST (exit $rc)"; STATUS=1
            fi
        else
            echo "[warn] no overview/metadata for $NAME - skipped highlight step for $PEST"
        fi
    done

    echo "=== $NAME done ($([ $STATUS -eq 0 ] && echo OK || echo "with errors")) -> $OUT_DIR/$NAME/ ==="
    return $STATUS
}

# ---- pick the image(s) (same sort order as the submit count) ---------------
mapfile -t SCANS < <(find "$SCANS_DIR" -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort)
if [ ${#SCANS[@]} -eq 0 ]; then echo "No JPEG/PNG images in $SCANS_DIR"; exit 1; fi

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    IMG="${SCANS[$SLURM_ARRAY_TASK_ID]:-}"
    if [ -z "$IMG" ]; then echo "No image for array index $SLURM_ARRAY_TASK_ID"; exit 1; fi
    process_one "$IMG"
else
    STATUS=0
    for IMG in "${SCANS[@]}"; do process_one "$IMG" || STATUS=1; done
    exit $STATUS
fi
