# Usage guide

## Training in detail

```bash
python train.py --dataset dataset --epochs 20 --batch-size 32 --augment
```

What happens, in order:

1. **Scan** — class folders and image counts are printed; the run stops
   early with a readable error if the dataset is missing, empty, has
   fewer than 2 classes, or any class has fewer than 2 images.
2. **Load** — images are decoded, resized to `--img-size` (default
   64×64) and normalised to `[0, 1]`. Corrupt files are skipped and
   counted, not fatal.
3. **Split** — stratified 80/20 train/test split (`--test-size`,
   `--random-state`), so every sign appears in both sets.
4. **Train** — Adam + sparse categorical cross-entropy, early stopping
   on validation loss (`--patience`, default 5; `0` disables),
   learning-rate reduction on plateau, optional augmentation layers
   with `--augment`.
5. **Save** — `model.keras`, `labels.json` and `training_history.json`
   side by side (paths configurable via `--model-output` /
   `--labels-output` / `$SLR_MODEL_PATH`).

Test accuracy is printed at the end, e.g.:

```text
INFO: Test accuracy: 93.50% (loss 0.2103)
INFO: Done. Model: model.keras | Labels: labels.json | Test accuracy: 93.50%
```

> `--img-size` is part of the model's contract: prediction must use the
> same size used in training (both default to 64).

## Prediction in detail

### Webcam (default)

```bash
python predict.py
python predict.py --source 1               # second camera
python predict.py --threshold 0.6 --padding 30
python predict.py --no-mediapipe           # classify the whole frame
```

MediaPipe finds the hand, a padded bounding box is cropped (always
clipped to the frame — a hand half out of view is skipped, not a
crash), the CNN classifies the crop, and the majority-voted label is
drawn on the frame with its confidence. Press **`q`** or **`Esc`** to
quit; `Ctrl+C` also exits cleanly and releases the camera.

### Video file

```bash
python predict.py --source recording.mp4
```

Same pipeline; the loop ends when the file does.

### Single image

```bash
python predict.py --image photo.jpg
# -> A (97%)

python predict.py --image photo.jpg --output annotated.jpg
```

Prints `label (confidence)` to stdout — `(?)` is appended when the
confidence is below `--threshold` — and optionally writes an annotated
copy. Needs no camera and no MediaPipe: ideal for testing a freshly
trained model (`tests/test_cli.py` exercises the failure paths of this
mode, and you can use it to smoke-test your own).

### Using a model stored elsewhere

```bash
python predict.py --model runs/v2/model.keras --labels runs/v2/labels.json
```

The predictor verifies that the number of labels matches the model's
output layer and refuses to start (with an explanation) when they come
from different training runs.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Expected error (missing dataset/model/camera, invalid options, ...) — message printed, no traceback |
| `130` | Interrupted with `Ctrl+C` |
