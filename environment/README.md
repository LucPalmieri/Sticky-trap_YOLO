# Environment

| File | For |
|---|---|
| `requirements-cpu.txt` | The classifier and post-processing scripts, without flatbug. Enough for the crop-level demo in `sample_data/pentastiridius/`. |
| `setup_flatbug_env.sh` | GPU conda env with flatbug + classifier deps, as used on the HPC (Tesla V100). |

For a CPU-only install that runs the **whole** pipeline (flatbug included),
follow the Installation section of the main README.

GPU notes (from production):
- On older GPUs (e.g. V100, compute capability 7.0) the default CUDA 13 torch
  wheel has no kernels; reinstall torch from the cu126 index (the script does
  this). `torch.cuda.is_available()` returning True does **not** prove the
  build works - run a real matmul on the GPU.
- If `fb_predict` fails with "Failed to find C compiler", install gcc or
  `export TORCHINDUCTOR_DISABLE=1 TORCHDYNAMO_DISABLE=1`.
