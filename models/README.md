# Pest model registry

One subfolder per target species classifier. `scripts/run_pipeline.sh` loops
over every subfolder here that has a `config.sh` and runs the
classify -> copy_positives -> highlight_positives chain once per pest, against
the *same* flatbug crops (detection is species-agnostic, so it only runs once
per trap image regardless of how many pests you're screening for).

```
models/
  _flatbug/
    flat_bug_M.pt        # detector weights (shared, not species-specific)
  pentastiridius/
    best.pt               # trained YOLO11s-cls classifier
    config.sh              # TARGET / MIN_CONF / WEIGHTS for this pest
  <new_pest>/
    <weights file>
    config.sh
```

## Adding a new pest

1. Create `models/<name>/`.
2. Drop the trained classifier weights in it.
3. Add a `config.sh` (see `models/pentastiridius/config.sh` for the template):
   ```bash
   TARGET="ClassNameInYourModel"   # must match a class name in the .pt file
   MIN_CONF="0.5"                  # below this top-1 confidence -> 'uncertain'
   WEIGHTS="your_weights_file.pt"  # relative to this folder
   ```
4. Nothing else changes. The next pipeline run picks it up automatically and
   writes its output under `<out_dir>/<image_name>/<name>/`.

Folders without a `config.sh` (like `_flatbug/`) are skipped by the loop, so
the shared detector weights live alongside the pest models without being
mistaken for one.

Model weights are not committed to this repository. Download the flatbug
weights into `_flatbug/` and train (or obtain) the classifier into
`pentastiridius/` - see the main README.
