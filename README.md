# Sign Language Recognition System

[![CI](https://github.com/OmkarSolat-17/Sign-Language-Recognition-System/actions/workflows/ci.yml/badge.svg)](https://github.com/OmkarSolat-17/Sign-Language-Recognition-System/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A real-time **Sign Language → Text** converter using Python, MediaPipe hand tracking, and a TensorFlow/Keras CNN — bridging communication barriers through AI-powered gesture recognition.

Point your webcam at your hand, show a sign, and the recognised label appears on screen with its confidence. The same model can also classify a single image or a pre-recorded video file, which makes it easy to test and demo without a camera.

> **⚠️ Read this first:** this repository contains **code only** — no pre-trained model and no dataset are bundled (they are large and dataset licences vary). You train your own model in minutes from a folder of images; see [Quick start](#-quick-start) and [docs/dataset.md](docs/dataset.md). Training and prediction therefore cannot work out-of-the-box until you provide a dataset — the tools below tell you exactly what's missing when you run them.

---

## Table of contents

- [Features](#-features)
- [How it works](#-how-it-works)
- [Quick start](#-quick-start)
- [Dataset](#-dataset)
- [Training](#-training)
- [Prediction](#-prediction)
- [CLI reference](#-cli-reference)
- [Project structure](#-project-structure)
- [Error handling](#-error-handling)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License & disclaimer](#-license--disclaimer)

## ✨ Features

- **Real-time webcam recognition** — MediaPipe detects and crops the hand; the CNN classifies the crop (~30 fps on a typical laptop CPU for this small model).
- **Three input modes** — webcam, video file (`--source clip.mp4`), and single image (`--image photo.jpg`, no camera needed).
- **Smoothed, confidence-aware output** — per-frame predictions are majority-vote smoothed so the on-screen text doesn't flicker, and guesses below the confidence threshold are visibly marked with `(?)`.
- **Train your own signs** — any gesture set works: one folder per sign (ASL letters, ISL words, your own vocabulary).
- **Reproducible labels** — class names are saved to `labels.json` next to the model at training time and validated against the model at prediction time. (The original version re-read a hard-coded dataset folder at prediction time, so renaming a folder silently corrupted every prediction — that bug is gone.)
- **Friendly failures everywhere** — missing dataset, corrupt images, missing model, busy camera, label/model mismatch: all produce one actionable `ERROR: ...` message and exit code 1, never a raw traceback. Corrupt images are skipped and counted, not fatal.
- **Fast, cheap tests** — the test suite runs in seconds *without* TensorFlow, a camera or a dataset.

## 🧠 How it works

```mermaid
flowchart LR
    A[Webcam / video / image] --> B{MediaPipe Hands}
    B -->|21 landmarks| C[Hand bounding box<br/>+ padding, clipped to frame]
    C --> D[Crop & resize to 64×64<br/>normalise to 0–1]
    D --> E[CNN<br/>Conv 32 → 64 → 128<br/>Dense 256 → softmax]
    E --> F[Argmax + confidence<br/>majority-vote smoothing]
    F --> G[Label drawn on frame<br/>e.g. “hello (94%)”]
```

Training mirrors the front half of the pipeline: every image in `dataset/<sign>/` is resized to 64×64, split into stratified train/test sets, and used to train the CNN defined in [`sign_language/model.py`](sign_language/model.py). See [docs/architecture.md](docs/architecture.md) for the full picture.

## 🚀 Quick start

### 1. Clone and install

```bash
git clone https://github.com/OmkarSolat-17/Sign-Language-Recognition-System.git
cd Sign-Language-Recognition-System

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt  # TensorFlow, OpenCV, MediaPipe, ...
```

> Python 3.10–3.12 is supported. If `pip install mediapipe` or `tensorflow` fails on a newer Python, use 3.11 — see [docs/troubleshooting.md](docs/troubleshooting.md).

### 2. Prepare a dataset

Create one folder per sign with images in it (full guide: [docs/dataset.md](docs/dataset.md)):

```text
dataset/
├── A/        *.jpg ...
├── B/        *.jpg ...
└── hello/    *.jpg ...
```

Validate it before training:

```bash
python scripts/check_dataset.py --dataset dataset
```

### 3. Train

```bash
python train.py --dataset dataset --epochs 20
```

This writes three files next to each other:

| File | What it is |
|---|---|
| `model.keras` | The trained model (use `--model-output` to change; legacy `.h5` output also works) |
| `labels.json` | The class names, in model-output order — **keep this with the model** |
| `training_history.json` | Per-epoch loss/accuracy, for plotting and debugging |

### 4. Predict

```bash
python predict.py                        # webcam — press q or Esc to quit
python predict.py --image photo.jpg      # single image, prints e.g. "A (97%)"
python predict.py --source clip.mp4      # video file
python predict.py --model path/to/model.keras --labels path/to/labels.json
```

## 📁 Dataset

The dataset is **not included** in this repository. You need at least **2 signs with at least 2 images each** (realistically: 100+ images per sign for useful accuracy). Practical ways to get one:

- Record your own with a webcam script/photo bursts — full control over signs and licensing, and the images match your camera.
- Use a public dataset such as the [ASL Alphabet dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — check its licence before redistributing, and note the folder layout maps directly onto `dataset/<sign>/`.

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. Corrupt or unreadable files are skipped with a warning and counted in the training summary. Full details, tips for balanced data, and the exact validation rules: **[docs/dataset.md](docs/dataset.md)**.

Alternative locations: `python train.py --dataset /path/to/data`, or set it once per shell:

```bash
export SLR_DATASET_PATH=/path/to/data   # Windows PowerShell: $env:SLR_DATASET_PATH = "C:\path\to\data"
```
(The old hard-coded `C:\Users\omkar\...` path is gone — no code path depends on one machine's filesystem any more.)

## 🏋️ Training

```bash
python train.py --dataset dataset --epochs 20 --batch-size 32 --augment
```

Notable behaviour:

- **Stratified split** (`--test-size 0.2` by default) — every class appears in both train and test sets; training refuses classes with < 2 images instead of crashing inside scikit-learn.
- **Early stopping + LR reduction** on validation loss (`--patience`, `0` disables early stopping).
- **Optional augmentation** (`--augment`): random flip/rotation/zoom layers, active only during training.
- **Everything is validated up front** — invalid values (e.g. `--epochs 0`, `--test-size 1.5`) fail immediately with a readable error.

## 🔮 Prediction

| Mode | Command | Needs camera? | Needs MediaPipe? |
|---|---|---|---|
| Webcam | `python predict.py` | ✅ | ✅ (or `--no-mediapipe`) |
| Video file | `python predict.py --source clip.mp4` | ❌ | ✅ (or `--no-mediapipe`) |
| Single image | `python predict.py --image photo.jpg --output annotated.jpg` | ❌ | ❌ |

- `--threshold 0.6` — predictions below this confidence display with `(?)` so uncertain guesses are never presented as fact.
- `--padding 30` — extra pixels around the detected hand box (the hand crop is always clipped to the frame; hands partially out of frame are skipped instead of crashing).
- `--no-mediapipe` — classify the whole frame; useful where MediaPipe can't be installed.
- The webcam loop survives transient frame failures but aborts with a clear error after 30 consecutive bad frames, and always releases the camera/windows on exit (`q`, `Esc`, `Ctrl+C`, or error). `model.predict` runs with `verbose=0` — no log spam.

## ⌨️ CLI reference

### `train.py`

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `$SLR_DATASET_PATH` or `./dataset` | Dataset folder (one sub-folder per sign) |
| `--model-output` | `./model.keras` | Output model file (`.keras` or `.h5`) |
| `--labels-output` | next to the model | Output `labels.json` path |
| `--img-size` | `64` | Square input size (train and predict must agree) |
| `--epochs` | `10` | Maximum training epochs |
| `--batch-size` | `32` | Training batch size |
| `--test-size` | `0.2` | Held-out fraction, strictly between 0 and 1 |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--patience` | `5` | Early-stopping patience (`0` = off) |
| `--random-state` | `42` | Seed for the split |
| `--augment` | off | Enable augmentation layers |
| `--verbose` | off | Debug logging |

### `predict.py`

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `$SLR_MODEL_PATH` or `./model.keras` | Trained model file |
| `--labels` | `labels.json` next to the model | Class labels file |
| `--image` | — | Classify one image and exit |
| `--output` | — | With `--image`: save an annotated copy |
| `--source` | `0` | Camera index or video file path |
| `--threshold` | `0.5` | Confidence threshold for the `(?)` marker |
| `--padding` | `20` | Hand-box padding in pixels |
| `--img-size` | `64` | Must match training |
| `--no-mediapipe` | off | Skip hand detection; classify whole frame |
| `--verbose` | off | Debug logging |

Exit codes: `0` success · `1` expected error (message printed) · `130` interrupted with `Ctrl+C`.

## 🗂 Project structure

```text
.
├── train.py / predict.py        # Thin CLI entry points (backwards compatible)
├── sign_language/               # The actual implementation
│   ├── config.py                # Validated TrainConfig / PredictConfig dataclasses
│   ├── data.py                  # Dataset discovery, loading, labels I/O
│   ├── model.py                 # CNN definition, save/load (lazy TensorFlow import)
│   ├── predictor.py             # SignPredictor: one image in → Prediction out
│   ├── training.py              # Training pipeline + train CLI
│   ├── inference.py             # Webcam/video/image pipeline + predict CLI
│   ├── utils.py                 # Bounding boxes, smoothing, logging, source parsing
│   └── exceptions.py            # SignLanguageError hierarchy (friendly failures)
├── scripts/check_dataset.py     # Validate a dataset without training
├── tests/                       # Pytest suite — no TF/camera/dataset needed
├── docs/                        # Architecture, dataset, training/prediction, troubleshooting
├── dataset/                     # Placeholder + guide (real datasets are git-ignored)
├── .github/                     # CI workflow, issue/PR templates, Dependabot
├── requirements.txt             # Runtime (ML) dependencies
├── requirements-dev.txt         # Test/lint dependencies (lightweight)
├── pyproject.toml               # Packaging, pytest and ruff config
├── Makefile                     # make test / lint / train / predict / ...
└── LICENSE                      # MIT
```

## 🛡 Error handling

Every failure a user can plausibly cause is a subclass of `SignLanguageError` with a message that says *what to do next*. Examples:

```text
$ python train.py --dataset ./nope
ERROR: Dataset directory not found: 'nope'. Pass --dataset <path>, set the SLR_DATASET_PATH
environment variable, or create a 'dataset/' folder with one sub-folder per sign (see docs/dataset.md).

$ python predict.py --image photo.jpg
ERROR: Labels file not found: 'labels.json'. It is written automatically by train.py next to
the model; pass --labels <path> if yours lives elsewhere, or retrain the model.
```

The full catalogue of errors, exit codes and design rules is in [docs/architecture.md](docs/architecture.md#error-handling-philosophy). The headline rules: never crash on one corrupt image, never leak a traceback for an expected failure, never present a low-confidence guess as certain, and always release the camera.

## 🛠 Development

```bash
pip install -r requirements-dev.txt   # pytest + ruff + numpy only — fast
make test                             # or: python -m pytest
make lint                             # or: python -m ruff check sign_language train.py predict.py scripts tests
make compile                          # syntax-check everything
```

Tests that need OpenCV/scikit-learn/TensorFlow skip automatically when those packages are absent, so the suite above passes on a bare checkout. The CI workflow (`.github/workflows/ci.yml`) runs lint + tests on Python 3.10–3.12 for every push and pull request.

## 🩺 Troubleshooting

Short version — see **[docs/troubleshooting.md](docs/troubleshooting.md)** for the long one:

- **`Dataset directory not found`** → pass `--dataset <path>` or set `SLR_DATASET_PATH`.
- **`Model file not found`** → train first (`python train.py ...`), or pass `--model <path>`.
- **Camera won't open** → close other apps using it, check OS camera permissions, try `--source 1`.
- **`pip install mediapipe/tensorflow` fails** → use Python 3.10–3.12 (3.11 is the safest bet) inside a virtual environment.
- **Predictions are one class off / nonsense after renaming folders** → your `labels.json` doesn't match the model; retrain or pass the matching `--labels`. The predictor refuses to run when the label *count* doesn't match the model, but only retraining fixes a same-count mismatch.

## 🗺 Roadmap

Ideas are tracked as GitHub issues; the most valuable next steps:

- [ ] Landmark-based model (classify MediaPipe's 21 hand landmarks instead of pixels — smaller, faster, lighting-invariant)
- [ ] Word/sentence assembly: stable-sign detection, space/delete gestures, on-screen sentence buffer
- [ ] A small dataset-recording tool (`scripts/record_dataset.py`) for collecting balanced data from a webcam
- [ ] Export to TFLite/ONNX for mobile and browser deployment
- [ ] Live demo (Streamlit/Gradio) for trying the model without a local install

## 🤝 Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup, coding standards and the pull-request process, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

## 📄 License & disclaimer

MIT — see [LICENSE](LICENSE).

This project is a **learning/research tool, not a certified translation aid**. Sign languages (ASL, ISL, BSL, ...) are complete languages with grammar far beyond single-sign classification; a model trained on a small image dataset will make mistakes, and predictions below the confidence threshold are explicitly marked `(?)` for that reason. Recognition accuracy depends heavily on your dataset's size, balance and similarity to your camera/lighting.
