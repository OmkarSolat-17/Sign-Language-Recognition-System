# Development guide

## Setup

```bash
git clone https://github.com/OmkarSolat-17/Sign-Language-Recognition-System.git
cd Sign-Language-Recognition-System
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # tests + lint (fast, no TensorFlow)
pip install -r requirements.txt       # only needed to actually train/predict
```

## Day-to-day commands

| Command | What it does |
|---|---|
| `make test` | Run the pytest suite |
| `make lint` | Ruff over all sources |
| `make compile` | Byte-compile everything (syntax smoke check) |
| `make check-dataset` | Validate `./dataset` (override with `DATASET=...`) |
| `make train` / `make predict` | Train / run webcam prediction |

## Testing philosophy

- The suite must pass **without** TensorFlow, OpenCV, MediaPipe, a
  camera or a dataset. Tests that genuinely need a heavy dependency use
  `pytest.importorskip(...)`, so they run in a full environment and
  skip in the lightweight one — exactly how CI is set up.
- CLI tests assert on **exit codes and stderr text**, because "fails
  with one clean line, no traceback" is a user-facing feature of this
  project, not an implementation detail (`tests/test_cli.py`).
- Dataset fixtures are built in `tmp_path`; scanning tests use empty
  files (scanning never decodes), decoding tests write real PNGs.
- Add tests with every bug fix: first a test that fails for the old
  reason, then the fix.

## Code conventions

- Python ≥ 3.10, 4-space indents, double quotes, ~100-column lines
  (`.editorconfig`, `pyproject.toml` / Ruff config).
- **No heavy imports at module top level** (TensorFlow, cv2, MediaPipe,
  sklearn): import inside functions and convert `ImportError` into
  `DependencyError` with the install command.
- **User-facing failures** raise `SignLanguageError` subclasses with an
  actionable message; library tracebacks must never reach the CLI.
- Entry points return exit codes (`0`/`1`/`130`), they don't call
  `sys.exit` deep in library code.
- Public functions get docstrings that state preconditions and raised
  exceptions; comments explain *why*, not *what*.

## Project layout rationale

`sign_language/` is the library; `train.py`/`predict.py` are thin
wrappers so the commands users already know keep working. New
capabilities belong in the package (with tests), not in the wrappers.
`scripts/` holds one-off utilities that reuse the package.
