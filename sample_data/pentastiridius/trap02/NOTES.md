# Sample: trap02

## Provenance
7 crops from one real trap scan processed during the 2026 field trial.
Traceable via the flatbug UUID (`132ac8e1-5a13-4d90-96d1-5858b1420d4e`) and
CROPNUMBERs to the original run. Not synthetic data.

The 7 crops are a curated subset of the 354 crops flatbug produced for this
trap: the single row the classifier flagged `TARGET` (Pentastiridius,
0.7287), plus 6 non-target crops spanning Beetle and Artefact. Smallest
positive count of the traps sampled so far - a good example of the common
"one candidate on this trap" case (most of the 27 field traps had 1-2 hits,
not the 4-5 seen in trap01/trap05).

## What this is for
Runs the classify stage with no GPU, no flatbug, and no HPC access:

```bash
python scripts/classify_crops.py \
  -w <best.pt> -i sample_data/pentastiridius/trap02/crops \
  -o out.csv --target Pentastiridius --min-conf 0.5 --device cpu
diff --strip-trailing-cr out.csv sample_data/pentastiridius/trap02/expected_classifications.csv
```

Labels should match exactly; confidences may differ in the last decimal
place depending on torch/ultralytics version. `expected_classifications.csv`
rows are in the same order `classify_crops.py` produces (sorted crop
filenames), so the `diff` is meaningful.

`positives_bench.csv` is the original `highlight_positives.py` output for
this trap (pixel coordinates of the 1 positive on the full scan) - included
for reference, not required to run the demo above.
