"""CNN model definition, saving and loading.

TensorFlow is imported lazily inside the functions so that ``--help``,
the dataset scanner and the unit tests all work on machines without
TensorFlow installed.  When it *is* needed but missing, users get a
:class:`~sign_language.exceptions.DependencyError` telling them the
exact install command instead of ``ModuleNotFoundError``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .exceptions import DependencyError, ModelError, ModelNotFoundError

logger = logging.getLogger(__name__)


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "TensorFlow is required for this operation but is not installed. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc
    return tf


def build_model(num_classes: int, img_size: int = 64, learning_rate: float = 1e-3,
                augment: bool = False):
    """Build and compile the sign-classification CNN.

    Same overall architecture as the original script (three
    convolution/pooling blocks followed by dense layers), hardened with
    input validation, dropout against overfitting, and the modern
    ``Input`` layer instead of the deprecated ``input_shape`` argument.

    Raises:
        ModelError: if ``num_classes`` or ``img_size`` are unusable.
        DependencyError: if TensorFlow is not installed.
    """
    if num_classes < 2:
        raise ModelError(
            f"A classifier needs at least 2 classes, got {num_classes}. "
            "Check that your dataset has at least two class sub-folders."
        )
    if img_size < 32:
        raise ModelError(
            f"img_size must be at least 32 for the three convolution/pooling "
            f"blocks in this architecture (smaller inputs shrink to a "
            f"zero-sized feature map), got {img_size!r}. Use --img-size 64 "
            "unless you have a reason not to."
        )
    if learning_rate <= 0:
        raise ModelError(f"learning_rate must be positive, got {learning_rate!r}.")

    tf = _require_tensorflow()
    layers, models = tf.keras.layers, tf.keras.models

    augmentation_layers = []
    if augment:
        # Inactive at inference time; only regularise training.
        augmentation_layers = [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]

    model = models.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            *augmentation_layers,
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="sign_language_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_model(model, model_path: str | Path) -> Path:
    """Save a trained model, creating parent folders as needed."""
    path = Path(model_path).expanduser()
    if path.suffix.lower() not in {".keras", ".h5"}:
        raise ModelError(
            f"model_path must end in '.keras' or '.h5', got '{path}'."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(path))
    except Exception as exc:
        raise ModelError(f"Failed to save model to '{path}': {exc}") from exc
    logger.info("Saved model to %s", path)
    return path


def load_model(model_path: str | Path):
    """Load a trained model (``.keras`` or legacy ``.h5``).

    Raises:
        ModelNotFoundError: the file does not exist — with a hint to
            train first.
        ModelError: the file exists but cannot be loaded.
        DependencyError: TensorFlow is not installed.
    """
    path = Path(model_path).expanduser()
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model file not found: '{path}'. Train a model first with "
            "'python train.py --dataset <path-to-dataset>', or pass "
            "--model <path> pointing at an existing .keras/.h5 file."
        )
    tf = _require_tensorflow()
    try:
        return tf.keras.models.load_model(str(path))
    except Exception as exc:
        raise ModelError(
            f"Could not load model '{path}': {exc}. The file may be "
            "corrupt or saved with an incompatible TensorFlow version — "
            "try retraining with train.py."
        ) from exc
