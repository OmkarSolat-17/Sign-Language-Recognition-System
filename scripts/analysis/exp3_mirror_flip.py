"""Experiment 3 — is ``--augment``'s horizontal flip safe for a gesture alphabet?

``sign_language/model.py`` adds ``RandomFlip("horizontal")`` whenever
``augment=True``.  A horizontal flip turns a right hand into a left hand and a
leftward sign into a rightward one.  If two of your classes differ only by
mirroring (handedness, direction, or an asymmetric finger arrangement), the
augmentation makes the two classes *the same image* half the time and the
accuracy ceiling collapses.

Run:  python scripts/analysis/exp3_mirror_flip.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from sign_language.model import build_model  # noqa: E402

IMG = 64


def mirror_pair_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Two classes: a symbol, and that symbol's exact horizontal mirror."""
    digits = load_digits()
    idx = np.flatnonzero(digits.target == 6)[:120]
    X, y = [], []
    for i in idx:
        symbol = cv2.resize((digits.images[i] * 16).astype(np.uint8), (IMG, IMG),
                            interpolation=cv2.INTER_NEAREST)
        X.append(cv2.cvtColor(symbol, cv2.COLOR_GRAY2BGR))
        y.append(0)
        X.append(cv2.cvtColor(cv2.flip(symbol, 1), cv2.COLOR_GRAY2BGR))
        y.append(1)
    return np.array(X, np.float32) / 255.0, np.array(y, np.int64)


def run(X, y, augment: bool, seed: int = 0) -> float:
    i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.25,
                                  random_state=seed, stratify=y)
    model = build_model(2, IMG, 1e-3, augment=augment)
    model.fit(X[i_tr], y[i_tr], epochs=40, batch_size=16, verbose=0, validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=6,
                                                          restore_best_weights=True)])
    return float(model.evaluate(X[i_te], y[i_te], verbose=0)[1])


def main() -> int:
    X, y = mirror_pair_dataset()
    print(f"mirror-pair dataset: {X.shape}, classes = symbol / its mirror")
    for augment in (False, True):
        accs = [run(X, y, augment, seed) for seed in (0, 1)]
        label = "WITH RandomFlip(horizontal)" if augment else "no flip"
        runs = ", ".join(f"{a*100:.1f}" for a in accs)
        print(f"  --augment {'ON ' if augment else 'OFF'}  {label:28s}: "
              f"{np.mean(accs)*100:6.2f}%   (runs: {runs})")
    print("\nConclusion: horizontal flip caps a mirror-sensitive alphabet at chance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
