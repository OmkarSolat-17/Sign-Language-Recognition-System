# Troubleshooting

## Installation

**`pip install tensorflow` / `mediapipe` fails or finds no wheel**
Use Python 3.10–3.12 (3.11 is the safest) in a virtual environment:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Very new Python releases often lack pre-built wheels for TensorFlow /
MediaPipe for a few months. `opencv-python` additionally needs system
GL libraries on some minimal Linux images
(`apt install libgl1` on Debian/Ubuntu).

**`ModuleNotFoundError: No module named 'tensorflow'` when running**
Either install the runtime deps (`pip install -r requirements.txt`) or
note that `--help`, the dataset checker and the unit tests are
*designed* to work without them — only actual training/prediction need
the ML stack, and those paths print the install command when it's
missing.

## Training

**`ERROR: Dataset directory not found: 'dataset'`**
The dataset isn't bundled. Create it (see
[dataset.md](dataset.md)), then pass `--dataset <path>` or set
`SLR_DATASET_PATH`.

**`ERROR: Each class needs at least 2 images ...`**
A class folder has 0–1 usable images. Add images, or remove the folder
if the sign isn't part of this model.

**Training prints `Skipping unreadable image ...`**
Those files are corrupt or not really images (e.g. an HTML error page
saved as `.jpg`). They're skipped and counted; training continues. If
*many* files skip, re-download/re-export the dataset.

**Accuracy is terrible**
Usually data, not code: too few images per sign, near-duplicate frames,
or test conditions (lighting/background/camera) unlike the training
images. Add varied data, try `--augment`, and confirm with
`--image` on photos that resemble your dataset before blaming the
webcam path. Also check you're predicting with the `labels.json` from
the same training run.

**Out of memory while loading**
`load_dataset` holds the (resized) dataset in RAM, which is fine for
typical sign datasets at 64×64 but not for tens of thousands of large
images. Reduce `--img-size`, trim the dataset, or move to a generator
based loader (roadmap item — contributions welcome).

## Prediction

**`ERROR: Model file not found: 'model.keras'`**
Train first (`python train.py --dataset ...`) or point `--model` at an
existing `.keras`/`.h5` file. Legacy `.h5` models from the original
script load fine — but they were trained without a `labels.json`, so
create one listing the class names **in the alphabetical folder order
used at training time**, or simply retrain.

**`ERROR: Label count (N) does not match the model's output size (M)`**
`labels.json` and the model come from different training runs. Retrain,
or pass the matching `--labels`.

**Predictions are consistently one sign off**
Same root cause as above, but with equal counts: the labels file lists
the right *number* of signs in the wrong order. Retrain so
`labels.json` is generated alongside the model — never hand-edit its
order afterwards.

**Camera won't open / black window**
Close other apps using the camera (Zoom, browser tabs, ...), check your
OS camera permission for the terminal/Python, try `--source 1`. On
Linux, confirm the device exists (`ls /dev/video*`). In remote/CI
environments there is no camera at all — use `--image` mode instead.

**`Failed to read 30 consecutive frames`**
The camera disconnected mid-stream or the video file is corrupt. The
loop aborts deliberately instead of spinning forever; reconnect and
rerun.

**Hand detected but nothing is drawn**
The hand's bounding box was empty after clipping (hand at the frame
edge) or every crop prediction failed — both are logged as warnings
with `--verbose`. Keep the hand fully in frame and reasonably lit.

**Predictions flicker between two signs**
Normal for similar signs: raise `--threshold` so uncertain output is
marked `(?)`, hold signs a little longer (the default smoother votes
over 5 frames), and add more training data for the confused pair.

## Still stuck?

Open an issue with: your OS and Python version, the exact command, the
full error output (add `--verbose`), and — for training problems — the
output of `python scripts/check_dataset.py --dataset <your path>`.
