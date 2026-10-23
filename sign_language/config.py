"""Validated configuration objects for training and prediction.

Configuration is intentionally kept in plain dataclasses so it can be
built from CLI arguments (see :mod:`sign_language.training` and
:mod:`sign_language.inference`), from environment variables, or directly
in Python code / notebooks.  All validation happens in ``__post_init__``
and raises :class:`~sign_language.exceptions.ConfigurationError` with a
message that tells the user exactly what to fix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError

#: Environment variable that can point at the dataset, so the old
#: hard-coded ``C:\\Users\\...`` path never has to come back.
ENV_DATASET_PATH = "SLR_DATASET_PATH"
ENV_MODEL_PATH = "SLR_MODEL_PATH"

DEFAULT_IMG_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SIZE = 0.2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_BBOX_PADDING = 20


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigurationError(f"{name} must be a positive number, got {value!r}.")


def _require_probability(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ConfigurationError(f"{name} must be between 0.0 and 1.0, got {value!r}.")


@dataclass
class TrainConfig:
    """Configuration for :func:`sign_language.training.run_training`."""

    dataset_path: Path = field(
        default_factory=lambda: Path(os.environ.get(ENV_DATASET_PATH, "dataset"))
    )
    model_path: Path = field(
        default_factory=lambda: Path(os.environ.get(ENV_MODEL_PATH, "model.keras"))
    )
    labels_path: Path | None = None  # defaults to ``labels.json`` next to the model
    img_size: int = DEFAULT_IMG_SIZE
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = 42
    learning_rate: float = DEFAULT_LEARNING_RATE
    augment: bool = False
    patience: int = 5

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path).expanduser()
        self.model_path = Path(self.model_path).expanduser()
        if self.labels_path is None:
            self.labels_path = self.model_path.with_name("labels.json")
        else:
            self.labels_path = Path(self.labels_path).expanduser()

        _require_positive("img_size", self.img_size)
        if self.img_size < 32:
            raise ConfigurationError(
                f"img_size must be at least 32 for this CNN architecture, "
                f"got {self.img_size!r}. Use --img-size 64 unless you have "
                "a reason not to."
            )
        _require_positive("epochs", self.epochs)
        _require_positive("batch_size", self.batch_size)
        _require_positive("learning_rate", self.learning_rate)
        if self.patience < 0:
            raise ConfigurationError(f"patience must be >= 0, got {self.patience!r}.")
        if not 0.0 < float(self.test_size) < 1.0:
            raise ConfigurationError(
                f"test_size must be between 0 and 1 (exclusive), got {self.test_size!r}."
            )
        if self.model_path.suffix.lower() not in {".keras", ".h5"}:
            raise ConfigurationError(
                "model_path must end in '.keras' (recommended) or '.h5' "
                f"(legacy), got '{self.model_path}'."
            )


@dataclass
class PredictConfig:
    """Configuration for :func:`sign_language.inference` entry points."""

    model_path: Path = field(
        default_factory=lambda: Path(os.environ.get(ENV_MODEL_PATH, "model.keras"))
    )
    labels_path: Path | None = None  # defaults to ``labels.json`` next to the model
    img_size: int = DEFAULT_IMG_SIZE
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    bbox_padding: int = DEFAULT_BBOX_PADDING
    camera_index: int = 0
    max_frame_failures: int = 30
    smoothing_window: int = 5

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path).expanduser()
        if self.labels_path is None:
            self.labels_path = self.model_path.with_name("labels.json")
        else:
            self.labels_path = Path(self.labels_path).expanduser()

        _require_positive("img_size", self.img_size)
        _require_probability("confidence_threshold", self.confidence_threshold)
        if self.bbox_padding < 0:
            raise ConfigurationError(
                f"bbox_padding must be >= 0, got {self.bbox_padding!r}."
            )
        if self.camera_index < 0:
            raise ConfigurationError(
                f"camera_index must be >= 0, got {self.camera_index!r}."
            )
        _require_positive("max_frame_failures", self.max_frame_failures)
        _require_positive("smoothing_window", self.smoothing_window)
