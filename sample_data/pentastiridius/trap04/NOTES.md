# Sample: trap04

## Provenance
11 crops from one real trap scan processed during the 2026 field trial.
Traceable via the flatbug UUID (`fb347136-e0df-440a-9981-379d46c6fd69`) and
CROPNUMBERs to the original run. Not synthetic data.

The 11 crops are a curated subset of the 387 crops flatbug produced for this
trap: all 3 rows the classifier flagged `TARGET` (Pentastiridius), plus 8
non-target crops spanning Fly, Artefact, and one sub-threshold
`uncertain`/`review` case.

Worth noting: this trap's 3 positives span a much wider confidence range
(0.53, 0.67, 0.97) than trap05's tight 0.74-0.99 cluster - a good
complementary example of the near-uniform 0.50-0.99 confidence spread
reported across the field trial (see "Status and limitations" in the main README).
No visual/expert read on true-vs-false-positive status for these three yet.

## What this is for
Runs the classify stage with no GPU, no flatbug, and no HPC access:

```bash
python scripts/classify_crops.py \
  -w <best.pt> -i sample_data/pentastiridius/trap04/crops \
  -o out.csv --target Pentastiridius --min-conf 0.5 --device cpu
diff --strip-trailing-cr out.csv sample_data/pentastiridius/trap04/expected_classifications.csv
```

Labels should match exactly; confidences may differ in the last decimal
place depending on torch/ultralytics version. `expected_classifications.csv`
rows are in the same order `classify_crops.py` produces (sorted crop
filenames), so the `diff` is meaningful.

`positives_bench.csv` is the original `highlight_positives.py` output for
this trap (pixel coordinates of the 3 positives on the full scan) - included
for reference, not required to run the demo above.
