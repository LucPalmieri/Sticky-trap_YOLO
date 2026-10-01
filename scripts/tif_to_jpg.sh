#!/bin/bash
# =============================================================================
# TIFF -> JPEG conversion for flatbug.
#
# Why: flatbug decodes images with torchvision.io.decode_image, which supports
# only jpeg/png/webp/gif. TIFF input fails with "Unsupported image file", so
# every .tif scan must be converted before it can enter the pipeline.
#
# One array task per TIFF (isolates a failure to a single file). CPU only -
# no GPU needed for a format conversion.
#
# SUBMIT (SLURM):
#   SD=/path/to/scans_tif
#   OD=/path/to/scans_jpg
#   N=$(find "$SD" -maxdepth 1 -type f \( -iname '*.tif' -o -iname '*.tiff' \) | wc -l)
#   mkdir -p logs
#   sbatch --array=0-$((N-1))%3 tif_to_jpg.sh "$SD" "$OD"
#
# RUN LOCALLY (no SLURM): converts every TIFF in <tif_dir> sequentially
#   bash tif_to_jpg.sh "$SD" "$OD"
#
# Environment: activates CONDA_ENV (conda) or VENV (venv) if set; otherwise
# uses whatever Python is already active. Only Pillow is needed.
# =============================================================================
# Set these for your cluster (account / partition are site-specific):
##SBATCH --account=<your_account>
##SBATCH --partition=<cpu_partition>
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=tif2jpg
#SBATCH --output=logs/tif2jpg_%A_%a.out
#SBATCH --error=logs/tif2jpg_%A_%a.err

set -uo pipefail

SRC_DIR=${1:?usage: sbatch tif_to_jpg.sh <tif_dir> <jpg_out_dir> [quality]}
DST_DIR=${2:?usage: sbatch tif_to_jpg.sh <tif_dir> <jpg_out_dir> [quality]}
QUALITY=${3:-95}

# ---- environment (optional; conda hook is required in batch mode) ---------
if [ -n "${CONDA_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"
    conda activate "$CONDA_ENV"
elif [ -n "${VENV:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

mkdir -p "$DST_DIR"

mapfile -t TIFS < <(find "$SRC_DIR" -maxdepth 1 -type f \
    \( -iname '*.tif' -o -iname '*.tiff' \) | sort)
if [ ${#TIFS[@]} -eq 0 ]; then echo "No TIFFs in $SRC_DIR"; exit 1; fi

convert_one() {
local IMG=$1
local BASE; BASE=$(basename "$IMG"); BASE="${BASE%.*}"
local OUT="$DST_DIR/$BASE.jpg"
echo "=== $IMG -> $OUT (Q=$QUALITY) ==="

if [ -s "$OUT" ]; then echo "already exists, skipping"; return 0; fi

python - "$IMG" "$OUT" "$QUALITY" <<'PY'
import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # these scans are far past the decompression-bomb guard
src, dst, q = sys.argv[1], sys.argv[2], int(sys.argv[3])

im = Image.open(src)
print(f"source: {im.size[0]}x{im.size[1]}  mode={im.mode}  "
      f"({im.size[0]*im.size[1]/1e6:.0f} MP)")
if im.mode != "RGB":
    # 16-bit / CMYK / paletted TIFFs must be reduced to 8-bit RGB for JPEG.
    # Note this in methods if the source was >8-bit: it is a real (if minor) loss.
    print(f"converting {im.mode} -> RGB (8-bit)")
    im = im.convert("RGB")
im.save(dst, "JPEG", quality=q, optimize=False)
print(f"wrote {dst}")
PY

if [ -s "$OUT" ]; then
    echo "=== done: $(du -h "$OUT" | cut -f1) ==="
else
    echo "conversion produced no output for $IMG"; return 1
fi
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # SLURM array mode: this task converts exactly one TIFF
    IMG="${TIFS[$SLURM_ARRAY_TASK_ID]:-}"
    if [ -z "$IMG" ]; then echo "No TIFF for array index $SLURM_ARRAY_TASK_ID"; exit 1; fi
    convert_one "$IMG"
else
    # local mode: convert them all, keep going past individual failures
    STATUS=0
    for IMG in "${TIFS[@]}"; do convert_one "$IMG" || STATUS=1; done
    exit $STATUS
fi
