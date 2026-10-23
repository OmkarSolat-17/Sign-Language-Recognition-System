"""A small, testable wrapper around a trained sign-classification model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .data import load_labels
from .exceptions import DependencyError, PredictionError
from .model import load_model
from .utils import check_labels_match_model

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """One model prediction."""

    label: str
    class_id: int
    confidence: float
    probabilities: list[float]
    threshold: float = 0.5

    @property
    def is_confident(self) -> bool:
        return self.confidence >= self.threshold


def preprocess_image(image, img_size: int = 64):
    """Resize (if needed) and normalise an image array to ``[0, 1]``.

    Works without OpenCV when the image already has the target size —
    resizing is the only step that needs it.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "NumPy is required but is not installed. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise PredictionError(
            f"Expected a colour image with shape (height, width, 3), "
            f"got shape {array.shape}."
        )
    if array.size == 0:
        raise PredictionError("Cannot predict on an empty image.")

    if array.shape[0] != img_size or array.shape[1] != img_size:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DependencyError(
                "OpenCV is required to resize images but is not installed. "
                "Install the project dependencies with: pip install -r requirements.txt"
            ) from exc
        array = cv2.resize(array, (img_size, img_size))

    array = array.astype("float32")
    if array.max() > 1.0:  # still in 0-255 range -> normalise
        array = array / 255.0
    return np.expand_dims(array, axis=0)


class SignPredictor:
    """Load a model + its labels once, then predict on image arrays.

    Raises the project's custom exceptions (never raw ``OSError`` /
    ``ValueError`` from deep inside TensorFlow) so CLI callers can show
    one clean error message.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path | None = None,
        img_size: int = 64,
        confidence_threshold: float = 0.5,
    ) -> None:
        if img_size <= 0:
            raise PredictionError(f"img_size must be positive, got {img_size!r}.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise PredictionError(
                f"confidence_threshold must be between 0 and 1, got {confidence_threshold!r}."
            )
        self.model_path = Path(model_path).expanduser()
        self.labels_path = (
            Path(labels_path).expanduser()
            if labels_path is not None
            else self.model_path.with_name("labels.json")
        )
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        self.labels: list[str] = load_labels(self.labels_path)
        self.model = load_model(self.model_path)
        try:
            output_units = int(self.model.output_shape[-1])
        except Exception as exc:
            raise PredictionError(
                f"Could not inspect the model's output shape: {exc}"
            ) from exc
        check_labels_match_model(self.labels, output_units)
        logger.info(
            "Loaded model '%s' with %d class(es): %s",
            self.model_path, len(self.labels), ", ".join(self.labels),
        )

    def predict(self, image) -> Prediction:
        """Predict the sign in one image array (``H x W x 3``)."""
        batch = preprocess_image(image, self.img_size)
        try:
            probabilities = self.model.predict(batch, verbose=0)
        except Exception as exc:
            raise PredictionError(f"Model prediction failed: {exc}") from exc

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise DependencyError(
                "NumPy is required but is not installed. "
                "Install the project dependencies with: pip install -r requirements.txt"
            ) from exc

        probs = np.asarray(probabilities, dtype=float)
        if probs.ndim != 2 or probs.shape[0] != 1 or probs.shape[1] != len(self.labels):
            raise PredictionError(
                f"Model returned an unexpected output shape {probs.shape}; "
                f"expected (1, {len(self.labels)})."
            )
        row = probs[0]
        class_id = int(np.argmax(row))
        return Prediction(
            label=self.labels[class_id],
            class_id=class_id,
            confidence=float(row[class_id]),
            probabilities=[float(p) for p in row],
            threshold=self.confidence_threshold,
        )
