# Sample: trap03

## Provenance
9 crops from one real trap scan processed during the 2026 field trial.
Traceable via the flatbug UUID (`fd243eb4-ce21-4a6d-98f4-ad1cb7879477`) and
CROPNUMBERs to the original run. Not synthetic data.

The 9 crops are a curated subset of the 214 crops flatbug produced for this
trap: the single row the classifier flagged `TARGET` (Pentastiridius,
0.7230), plus 8 non-target crops - including an `Aphidoidea` call and a
sub-threshold `uncertain`/`review` case, giving this sample the widest
label variety of the traps included so far.

## What this is for
Runs the classify stage with no GPU, no flatbug, and no HPC access:

```bash
python scripts/classify_crops.py \
  -w <best.pt> -i sample_data/pentastiridius/trap03/crops \
  -o out.csv --target Pentastiridius --min-conf 0.5 --device cpu
diff --strip-trailing-cr out.csv sample_data/pentastiridius/trap03/expected_classifications.csv
```

Labels should match exactly; confidences may differ in the last decimal
place depending on torch/ultralytics version. `expected_classifications.csv`
rows are in the same order `classify_crops.py` produces (sorted crop
filenames), so the `diff` is meaningful.

`positives_bench.csv` is the original `highlight_positives.py` output for
this trap (pixel coordinates of the 1 positive on the full scan) - included
for reference, not required to run the demo above.
