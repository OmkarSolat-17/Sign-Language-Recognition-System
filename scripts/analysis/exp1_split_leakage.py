"""Experiment 1 — how optimistic is the number train.py prints?

``sign_language/training.py`` splits the data once into train/test and then
passes the *test* set as ``validation_data``.  EarlyStopping picks the epoch
that scores best on that set, and the accuracy printed afterwards is measured
on the very same images.  The metric is therefore tuned on itself.

This script trains the repo's model twice on the same data:

* "repo pipeline"    — train/test split, test doubles as validation
* "honest pipeline"  — train/validation/test, test never seen during training

Run:  python scripts/analysis/exp1_split_leakage.py [--small]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from sign_language.data import split_dataset  # noqa: E402
from sign_language.model import build_model  # noqa: E402

IMG = 64
N_CLASSES = 10


def stand_in_dataset(per_class: int | None) -> tuple[np.ndarray, np.ndarray]:
    """A 10-class symbol set used as a stand-in for a gesture alphabet.

    scikit-learn's digits set is the closest small, real, many-class image
    problem available offline: 8x8 bitmaps, 1797 samples.  Upscaled to the
    repo's default 64x64 and replicated to 3 channels.
    """
    digits = load_digits()
    X = np.stack(
        [
            np.repeat(
                tf.image.resize(img[..., None].astype("float32") / 16.0, (IMG, IMG)).numpy(),
                3,
                axis=-1,
            )
            for img in digits.images
        ]
    )
    y = digits.target.astype(np.int64)
    if per_class:
        rng = np.random.default_rng(0)
        keep = [rng.choice(np.flatnonzero(y == c), size=per_class, replace=False)
                for c in np.unique(y)]
        keep = np.concatenate(keep)
        X, y = X[keep], y[keep]
    return X, y


def _callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]


def repo_pipeline(X, y, seed: int) -> float:
    X_tr, X_te, y_tr, y_te = split_dataset(X, y, test_size=0.2, random_state=seed)
    model = build_model(N_CLASSES, IMG, 1e-3, augment=False)
    model.fit(X_tr, y_tr, epochs=30, batch_size=32,
              validation_data=(X_te, y_te),          # <-- the leak
              callbacks=_callbacks(), verbose=0)
    return float(model.evaluate(X_te, y_te, verbose=0)[1])


def honest_pipeline(X, y, seed: int) -> float:
    X_tr, X_tmp, y_tr, y_tmp = split_dataset(X, y, test_size=0.3, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp
    )
    model = build_model(N_CLASSES, IMG, 1e-3, augment=False)
    model.fit(X_tr, y_tr, epochs=30, batch_size=32,
              validation_data=(X_val, y_val),
              callbacks=_callbacks(), verbose=0)
    return float(model.evaluate(X_te, y_te, verbose=0)[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true",
                        help="25 images per class, i.e. what a hand-collected sign set "
                             "looks like (this is where the gap becomes large).")
    args = parser.parse_args()

    X, y = stand_in_dataset(25 if args.small else None)
    print(f"stand-in dataset: {X.shape}, {len(np.unique(y))} classes")

    reported, honest = [], []
    for seed in (0, 1, 2):
        r, h = repo_pipeline(X, y, seed), honest_pipeline(X, y, seed)
        reported.append(r)
        honest.append(h)
        print(f"  seed {seed}: repo-reported={r*100:6.2f}%   honest held-out={h*100:6.2f}%")

    print(f"\nMEAN repo-reported  : {np.mean(reported)*100:6.2f}%")
    print(f"MEAN honest held-out: {np.mean(honest)*100:6.2f}%")
    print(f"optimistic bias     : {(np.mean(reported)-np.mean(honest))*100:+.2f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
