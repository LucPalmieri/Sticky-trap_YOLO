# Sample: trap05

## Provenance
13 crops from one real trap scan processed during the 2026 field trial.
Traceable via the flatbug UUID (`360f0c4c-df56-4d71-b083-25bc85a6313d`) and
CROPNUMBERs to the original run. Not synthetic data.

The 13 crops are a curated subset of the 316 crops flatbug produced for this
trap: the 5 rows the classifier flagged `TARGET` (Pentastiridius, confidence
0.74-0.99), plus 8 non-target crops spanning Artefact, Fly, Other insects,
and one sub-threshold `uncertain`/`review` case - chosen to show the
classifier's full range of outputs, not just its hits.

## What this is for
Runs the classify stage with no GPU, no flatbug, and no HPC access:

```bash
python scripts/classify_crops.py \
  -w <best.pt> -i sample_data/pentastiridius/trap05/crops \
  -o out.csv --target Pentastiridius --min-conf 0.5 --device cpu
diff --strip-trailing-cr out.csv sample_data/pentastiridius/trap05/expected_classifications.csv
```

Labels should match exactly; confidences may differ in the last decimal
place depending on torch/ultralytics version. `expected_classifications.csv`
rows are in the same order `classify_crops.py` produces (sorted crop
filenames), so the `diff` is meaningful.

`positives_bench.csv` is the original `highlight_positives.py` output for
this trap (pixel coordinates of the 5 positives on the full scan) - included
for reference, not required to run the demo above.

## Caveat - read before using this as a "the model works" example
The 5 `TARGET` rows here were visually reviewed by the project author, an
entomologist, and are suspected to be false positives - not confirmed
either way by COI barcoding. This is not a failure being hidden: it is a
direct, real illustration of the project's own rule that a `TARGET` flag is
a candidate for lab confirmation, not an identification (see the main
README's limitations section). If anything, this is the more honest sample
to lead with precisely because it isn't a clean win.
