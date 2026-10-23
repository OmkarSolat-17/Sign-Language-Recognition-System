# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-10-23

First structured release: the original two-script prototype rebuilt as
a tested, documented package. Existing workflows
(`python train.py`, `python predict.py`) keep working.

### Added

- `sign_language/` package: `config`, `data`, `model`, `predictor`,
  `training`, `inference`, `utils`, `exceptions` modules.
- CLI flags for both scripts (`--dataset`, `--epochs`, `--model-output`,
  `--image`, `--source`, `--threshold`, `--padding`, `--no-mediapipe`,
  `--augment`, `--verbose`, ...), plus `SLR_DATASET_PATH` /
  `SLR_MODEL_PATH` environment variables.
- `labels.json` written at training time and validated against the
  model at prediction time; `training_history.json` saved per run.
- Single-image and video-file prediction modes (no camera required).
- Confidence display with a `(?)` marker below `--threshold`;
  majority-vote prediction smoothing to stop label flicker.
- Proper error handling throughout: a `SignLanguageError` hierarchy,
  actionable `ERROR: ...` messages, exit codes 0/1/130, corrupt images
  skipped and counted, stratified-split preconditions checked up front,
  camera/MediaPipe resources always released.
- Stratified train/test split, early stopping, LR-on-plateau,
  optional data augmentation, dropout in the CNN.
- `scripts/check_dataset.py` dataset validator.
- Pytest suite (60 tests) that runs without TensorFlow/OpenCV/a
  camera/a dataset; GitHub Actions CI on Python 3.10–3.12; Ruff lint.
- Documentation: README rewrite, `docs/architecture.md`,
  `docs/dataset.md`, `docs/usage.md`, `docs/troubleshooting.md`,
  `docs/development.md`.
- Project scaffolding: `pyproject.toml`, `requirements.txt`,
  `requirements-dev.txt`, `Makefile`, `.gitignore`, `.editorconfig`,
  MIT `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, issue/PR templates, Dependabot.

### Fixed

- Hard-coded Windows dataset path (`C:\Users\omkar\...`) in both
  scripts — the cause of "works on my machine" failures for everyone
  else, and of silently wrong labels at prediction time.
- Crash on the first corrupt/undecodable image during training
  (uncaught `cv2.resize(None)` failure).
- Crash on empty hand crops when the hand touched the frame edge.
- `model.predict` log spam on every frame; camera and windows not
  released on error paths; infinite loop on a dead camera.

### Changed

- Default model format is now `.keras` (legacy `.h5` still loads).
- Class names at prediction time come from `labels.json`, **not** from
  re-scanning a dataset folder.

## [0.1.0] — pre-release prototype

- Original `train.py` (folder dataset → CNN → `model.h5`) and
  `predict.py` (webcam + MediaPipe crop → CNN) uploaded as a prototype.
