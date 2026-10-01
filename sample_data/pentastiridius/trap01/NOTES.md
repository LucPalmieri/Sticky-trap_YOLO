# Sample: trap01

## Provenance
11 crops from one real trap scan processed during the 2026 field trial.
Traceable via the flatbug UUID (`c982427f-531c-4271-8a7b-11bf73ab12fd`) and
CROPNUMBERs to the original run. Not synthetic data.

The 11 crops are a curated subset of the 240 crops flatbug produced for this
trap: all 4 rows the classifier flagged `TARGET` (Pentastiridius, confidence
0.51-0.71 - the tightest, lowest-confidence cluster of the traps sampled so
far), plus 7 non-target crops spanning Artefact and Fly.

## What this is for
Runs the classify stage with no GPU, no flatbug, and no HPC access:

```bash
python scripts/classify_crops.py \
  -w <best.pt> -i sample_data/pentastiridius/trap01/crops \
  -o out.csv --target Pentastiridius --min-conf 0.5 --device cpu
diff --strip-trailing-cr out.csv sample_data/pentastiridius/trap01/expected_classifications.csv
```

Labels should match exactly; confidences may differ in the last decimal
place depending on torch/ultralytics version. `expected_classifications.csv`
rows are in the same order `classify_crops.py` produces (sorted crop
filenames), so the `diff` is meaningful.

`positives_bench.csv` is the original `highlight_positives.py` output for
this trap (pixel coordinates of the 4 positives on the full scan) - included
for reference, not required to run the demo above.
