#!/usr/bin/env python3
"""Train the sign-language recognition model.

This is a thin, backwards-compatible entry point — the implementation
lives in :mod:`sign_language.training`.  Run ``python train.py --help``
for all options.

Quick start::

    python train.py --dataset dataset --epochs 20

On success this writes ``model.keras``, ``labels.json`` and
``training_history.json`` (all paths configurable; see ``--help``).
Expected failures (missing dataset, too few images, missing
dependencies, ...) print a single actionable ``ERROR: ...`` line and
exit with status 1 instead of a traceback.
"""

import sys

from sign_language.training import main

if __name__ == "__main__":
    sys.exit(main())
