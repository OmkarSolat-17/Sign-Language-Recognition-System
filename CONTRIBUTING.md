# Contributing

Thanks for helping make sign-language recognition more accessible! 🎉

## Getting started

1. Fork the repo and clone your fork.
2. Create a branch: `git checkout -b my-change`.
3. Set up the dev environment (tests run *without* the heavy ML stack):

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```

4. Make your change, add tests, and update docs if behaviour changes.
5. Verify: `make test && make lint && make compile`.
6. Open a pull request (see below).

Full details: [docs/development.md](docs/development.md).

## What makes a good contribution

- **Bug fixes** with a regression test that fails before your fix.
- **Error-handling improvements** — a confusing failure turned into a
  clear, actionable message is a first-class contribution here.
- **Dataset/training quality** — better augmentation, a landmark-based
  model, an on-disk data loader for large datasets.
- **Docs** — especially troubleshooting entries from real failures you hit.

Please keep pull requests focused: one logical change per PR, with a
description of *what* changed, *why*, and how you tested it.

## Coding standards

- Python ≥ 3.10; follow the Ruff config in `pyproject.toml`
  (`make lint` must pass) and the conventions in
  [docs/development.md](docs/development.md#code-conventions) — most
  importantly: no top-level TensorFlow/OpenCV/MediaPipe imports, and
  user-facing failures raise `SignLanguageError` subclasses.
- New behaviour needs tests; the suite must pass without TensorFlow, a
  camera, or a dataset.
- Don't commit datasets, trained models, or `labels.json` — they're
  git-ignored build artefacts.

## Reporting bugs

Use the bug-report template and include: OS, Python version, the exact
command, the full output (with `--verbose`), and for training issues
the output of `python scripts/check_dataset.py --dataset <path>`.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
