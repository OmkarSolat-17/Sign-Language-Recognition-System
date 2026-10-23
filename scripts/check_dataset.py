#!/usr/bin/env python3
"""Validate a dataset folder without training anything.

Usage::

    python scripts/check_dataset.py --dataset dataset

Prints the per-class image counts and exits 0 when the dataset looks
trainable (>= 2 classes, >= 2 images per class), 1 otherwise.  Handy
in CI and before starting a long training run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sign_language.data import scan_dataset  # noqa: E402
from sign_language.exceptions import SignLanguageError  # noqa: E402
from sign_language.utils import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="dataset", help="Dataset folder (default: ./dataset).")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        summary = scan_dataset(args.dataset)
    except SignLanguageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset: {summary.root}")
    print(f"Classes: {summary.num_classes} | Images: {summary.total_images}")
    problems = []
    for name in summary.classes:
        count = summary.counts[name]
        flag = "" if count >= 2 else "  <-- too few for a train/test split"
        if count < 2:
            problems.append(name)
        print(f"  {name:<20} {count:>6} image(s){flag}")
    if summary.num_classes < 2:
        print("ERROR: at least 2 classes are required.", file=sys.stderr)
        return 1
    if problems:
        print(f"ERROR: class(es) with < 2 images: {', '.join(problems)}", file=sys.stderr)
        return 1
    print("Dataset looks trainable. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
