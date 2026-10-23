# Dataset guide

## Layout

One sub-folder per sign, images directly inside it:

```text
dataset/
├── A/
│   ├── img001.jpg
│   └── ...
├── B/
├── hello/
└── thank_you/
```

Rules enforced by `sign_language.data.scan_dataset`:

- The dataset root must exist and be a directory.
- At least one class folder is needed to scan; **training requires ≥ 2
  classes and ≥ 2 images per class** (the stratified train/test split
  needs both splits to contain every class — ideally use far more).
- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
  (case-insensitive).
- Files at the dataset root, hidden files/folders (`.something`) and
  unsupported extensions are ignored with a warning, never an error.
- Corrupt/undecodable images are skipped during loading, counted, and
  reported in the training summary (`skipped N unreadable file(s)`).
- Class order is alphabetical and frozen into `labels.json` at training
  time — after training, renaming dataset folders does **not** affect
  prediction (and must not be expected to).

## Checking a dataset

```bash
python scripts/check_dataset.py --dataset dataset
python train.py --dataset /path/to/data     # or:
export SLR_DATASET_PATH=/path/to/data
```

`check_dataset.py` prints per-class counts and exits non-zero when the
dataset isn't trainable yet.

## How much data?

Very rough guidance for the pixel-CNN in this repo:

| Images per sign | What to expect |
|---|---|
| 2–20 | Pipeline smoke test only; the model will memorise, not learn |
| 100–300 | Usable demo for a handful of distinct signs, same person/lighting |
| 500+ per sign, multiple people | Starts to generalise across hands/lighting/backgrounds |

Balance matters more than raw count: similar numbers of images per
sign, several hand positions/angles/distances, and backgrounds similar
to where you'll run prediction. Beware near-duplicate bursts (30
consecutive webcam frames are effectively one sample).

## Getting a dataset

- **Record your own** — best licence situation and best match to your
  camera. (A recording helper is on the roadmap; meanwhile any
  webcam photo tool works — save bursts into `dataset/<sign>/`.)
- **Public datasets** — e.g. the ASL Alphabet dataset on Kaggle. Check
  the licence before redistributing images, and do not commit datasets
  to this repository (`dataset/` is git-ignored for that reason).

## What is *not* in git

No dataset and no trained model are committed. `model.keras`,
`labels.json` and `training_history.json` are build artefacts: anyone
can reproduce them with `train.py`, and they belong in release assets
or model storage, not in source control.
