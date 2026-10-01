#!/bin/bash
# =============================================================================
# GPU environment for flatbug + the full pipeline (run_pipeline.sh).
#
# Lines marked [VERIFIED] come from the production HPC setup (Tesla V100);
# lines marked [RECONSTRUCTED] are the standard way to do a step that was
# originally done by hand. For a CPU-only install, see the main README.
#
# Usage:
#   bash environment/setup_flatbug_env.sh <conda_env_path>
# =============================================================================
set -euo pipefail

ENV_PATH=${1:?usage: bash setup_flatbug_env.sh <conda_env_path>}
FLATBUG_DIR=${FLATBUG_DIR:-./flat-bug}

# [VERIFIED] Python 3.11 - confirmed from the installed package paths seen in
# real flatbug tracebacks (envs/flatbug/lib/python3.11/site-packages/...).
conda create -y -p "$ENV_PATH" python=3.11
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

# [RECONSTRUCTED] flat-bug is installed from a source clone, not PyPI.
if [ ! -d "$FLATBUG_DIR" ]; then
    git clone https://github.com/darsa-group/flat-bug.git "$FLATBUG_DIR"
fi
pip install -e "$FLATBUG_DIR"

# [VERIFIED] the V100 (compute capability 7.0) has no kernels for the cu130
# wheel that flat-bug's own dependency resolution pulls in by default.
# torch.cuda.is_available() returns True either way - only a real kernel
# launch reveals the mismatch ("no kernel image is available for execution
# on the device"). --force-reinstall is mandatory: pip silently no-ops
# when the installed version string already matches.
pip install torch==2.13.0 torchvision==0.28.0 \
    --index-url https://download.pytorch.org/whl/cu126 --force-reinstall

# Classifier deps. flat-bug pins ultralytics<=8.3.124, which is fine for
# inference (the full pipeline was verified end-to-end on 8.3.124). The
# classifier was *trained* on 8.4.98; to reproduce training exactly, use a
# separate env with `pip install ultralytics==8.4.98`.
pip install pandas scikit-learn

# [VERIFIED] flatbug's NMS path uses torch.compile via Triton, which needs a
# C compiler on PATH. Install one (e.g. `apt install gcc` / `conda install
# gcc`) OR disable compilation:
#   export TORCHINDUCTOR_DISABLE=1 TORCHDYNAMO_DISABLE=1
echo "If fb_predict fails with 'Failed to find C compiler', either install"
echo "gcc or export TORCHINDUCTOR_DISABLE=1 TORCHDYNAMO_DISABLE=1"

# [VERIFIED, HPC-specific] only needed if your /home has a disk quota -
# redirects conda/pip caches and tmp to a larger volume. Skip on a
# workstation with no such quota.
if [ -n "${FLATBUG_CACHE_ROOT:-}" ]; then
    export CONDA_PKGS_DIRS="$FLATBUG_CACHE_ROOT/.conda/pkgs"
    export CONDA_ENVS_DIRS="$FLATBUG_CACHE_ROOT/.conda/envs"
    export PIP_CACHE_DIR="$FLATBUG_CACHE_ROOT/.cache/pip"
    export TMPDIR="$FLATBUG_CACHE_ROOT/tmp"
fi

echo ""
echo "Setup done. Verify the GPU actually works (torch.cuda.is_available()"
echo "is NOT sufficient - it returns True even with the wrong CUDA build):"
echo '  python -c "import torch; a=torch.randn(8,8,device
='"'"'cuda'"'"'); print((a@a).sum())"'
