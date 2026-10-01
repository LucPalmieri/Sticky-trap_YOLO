# Sample: trap06

## Provenance
11 crops from one real trap scan processed during the 2026 field trial.
Traceable via the flatbug UUID (`d11fd82d-2464-41e6-a19b-0c1259eec050`) and
CROPNUMBERs to the original run. Not synthetic data.

The 11 crops are a curated subset of the 452 crops flatbug produced for this
trap (the largest crop count of any trap sampled so far): all 3 rows the
classifier flagged `TARGET` (Pentastiridius, confidence 0.62-0.93), plus 8
non-target crops spanning Fly, Artefact, Beetle, Other insects, and one
sub-threshold `uncertain`/`review` case.

Note: `scan_samples/Pentastiridius/crops_demo/trap06/` also contains a
`trap06.jpg` full-resolution scan (107MB) that doesn't match the crop
filename convention and wasn't pulled into this sample - flagged separately,
not included here.

## What this is for
Runs the classify stage with no GPU, no flatbug, and no HPC access:

```bash
python scripts/classify_crops.py \
  -w <best.pt> -i sample_data/pentastiridius/trap06/crops \
  -o out.csv --target Pentastiridius --min-conf 0.5 --device cpu
diff --strip-trailing-cr out.csv sample_data/pentastiridius/trap06/expected_classifications.csv
```

Labels should match exactly; confidences may differ in the last decimal
place depending on torch/ultralytics version. `expected_classifications.csv`
rows are in the same order `classify_crops.py` produces (sorted crop
filenames), so the `diff` is meaningful.

`positives_bench.csv` is the original `highlight_positives.py` output for
this trap (pixel coordinates of the 3 positives on the full scan) - included
for reference, not required to run the demo above.
