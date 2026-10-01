# Tier 2 sample: full-pipeline / batch-mode demo

**Needs flatbug installed** (GPU recommended; CPU works, ~minutes per image) (unlike `sample_data/pentastiridius/`,
which needs neither). This tier exercises `scripts/run_pipeline.sh`
end-to-end - flatbug detection, classification, positive extraction, and
bench localisation - and specifically the SLURM array / per-image isolation
design, not just the classifier.

## What these are
6 real trap scans, each a 3000x3000px **native-resolution** crop (no
downscaling - see `sample_data/make_scan_samples.py`) taken from one of the
6 full field scans (not included - ~100 MB+ each), centred on that trap's
highest-confidence Pentastiridius candidate from the real 2026 field trial.
`manifest.csv` records, per file, which crop it came from, the anchor
candidate's original CROPNUMBER/confidence, and its pixel position - **by
construction, the anchor sits at the exact centre of every crop, (1500,1500)**.

## Running it
```bash
# SLURM (from the repo root)
mkdir -p logs
sbatch --array=0-5 scripts/run_pipeline.sh sample_data/scans_demo sample_data/scans_demo/out
# or locally, no SLURM
DEVICE=cpu bash scripts/run_pipeline.sh sample_data/scans_demo sample_data/scans_demo/out
```
This submits one SLURM array task per image - the same pattern used for the
real 27-trap field batches, just smaller. Each task runs flatbug fresh on
its 9-megapixel crop, then every configured pest model in `models/`.

## What to expect (and what not to)
- flatbug will produce its own new crop numbering for each image - it is
  re-detecting from scratch on a cropped, lower-context image, so the
  original CROPNUMBERs in `manifest.csv` will **not** reappear. Use it only
  to sanity-check that *a* Pentastiridius candidate turns up near the
  crop's centre, not to reproduce the original crop-index provenance.
- Fewer total detections per image than the full trap (this is a
  9-megapixel window of an ~800-megapixel scan, so most of the trap's other
  insects aren't in frame).
- A confident re-detection here is a reasonable sanity check that your
  flatbug + classifier install behaves as expected on unfamiliar hardware;
  it is not a validation of the classifier (see "Status and limitations"
  in the main README).

## Regenerating this folder
`sample_data/make_scan_samples.py` built it from the full-resolution scans
(not included) - kept for provenance, not meant for reuse.

## Reference run (2026-10-01, CPU, ultralytics 8.3.124)
`DEVICE=cpu bash scripts/run_pipeline.sh sample_data/scans_demo out/` finished
all 6 images. In 5 of 6 a `TARGET` candidate was flagged within ~110 px of
the centre (conf 0.57-0.89). In trap05 the centre insect was re-detected
but classified `Beetle` (0.82), whereas the same insect cropped from the full
scan was `Pentastiridius` (0.99). The call depends on how the insect is
cropped, which is one more reason to treat flags as candidates only.
