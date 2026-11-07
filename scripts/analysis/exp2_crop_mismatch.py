"""Experiment 2 — training and prediction preprocess images differently.

* ``sign_language/data.py::load_dataset``   resizes the **whole image** to 64x64.
* ``sign_language/inference.py::run_stream`` crops the **MediaPipe hand box**
  and only then resizes.

If the training images are webcam-style frames (a hand somewhere in the
frame), the model learns "hand occupying ~30% of the canvas" and is then asked
about "hand filling the canvas".  Same classes, different distribution.

Run:  python scripts/analysis/exp2_crop_mismatch.py
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
FRAME = 160
GLYPH = 48          # symbol occupies ~9% of the frame area, like a hand at arm's length


def build_frames():
    """Return (full frames, tight MediaPipe-style crops, labels)."""
    digits = load_digits()
    rng = np.random.default_rng(0)
    frames, crops, labels = [], [], []
    for img, label in zip(digits.images, digits.target, strict=True):
        glyph = cv2.resize((img * 16).astype(np.uint8), (GLYPH, GLYPH),
                           interpolation=cv2.INTER_NEAREST)
        glyph = cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)
        frame = np.zeros((FRAME, FRAME, 3), np.uint8)
        ox, oy = rng.integers(10, FRAME - GLYPH - 10, 2)
        frame[oy:oy + GLYPH, ox:ox + GLYPH] = glyph
        noise = rng.normal(0, 12, frame.shape).astype(np.int16).clip(0, 255).astype(np.uint8)
        frame = cv2.add(frame, noise)
        frames.append(cv2.resize(frame, (IMG, IMG)))                        # repo training path
        crops.append(cv2.resize(frame[oy:oy + GLYPH, ox:ox + GLYPH], (IMG, IMG)))  # inference path
        labels.append(label)
    return (np.array(frames, np.float32) / 255.0,
            np.array(crops, np.float32) / 255.0,
            np.array(labels, np.int64))


def run(train_on, test_on, y, seed: int = 0, epochs: int = 25) -> float:
    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=0.25, random_state=seed, stratify=y)
    model = build_model(10, IMG, 1e-3, augment=False)
    model.fit(train_on[i_tr], y[i_tr], epochs=epochs, batch_size=32, verbose=0,
              validation_split=0.1,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=4,
                                                          restore_best_weights=True)])
    return float(model.evaluate(test_on[i_te], y[i_te], verbose=0)[1])


def main() -> int:
    X_full, X_crop, y = build_frames()
    print(f"frames {X_full.shape} (symbol ~9% of canvas) | crops {X_crop.shape} (symbol fills canvas)")
    print("chance level for 10 classes: 10.00%\n")

    offline = run(X_full, X_full, y)
    live = run(X_full, X_crop, y)
    consistent = run(X_crop, X_crop, y)

    print(f"train full-frame -> test full-frame : {offline*100:6.2f}%   the number train.py reports")
    print(f"train full-frame -> test hand crop  : {live*100:6.2f}%   <-- the live webcam path")
    print(f"train hand crop  -> test hand crop  : {consistent*100:6.2f}%   consistent preprocessing")
    print(f"\naccuracy lost to the mismatch       : {(offline-live)*100:+.2f} points")
    print(f"accuracy recovered by consistency   : {(consistent-live)*100:+.2f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
