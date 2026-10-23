"""Sign Language Recognition System.

A real-time sign-language-to-text system: MediaPipe locates the hand,
a small CNN (TensorFlow/Keras) classifies the hand crop, and the label
is shown as text.  See the project README for the full guide.
"""

from .exceptions import (
    CameraError,
    ConfigurationError,
    DatasetError,
    DatasetNotFoundError,
    DependencyError,
    LabelsError,
    ModelError,
    ModelNotFoundError,
    PredictionError,
    SignLanguageError,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "CameraError",
    "ConfigurationError",
    "DatasetError",
    "DatasetNotFoundError",
    "DependencyError",
    "LabelsError",
    "ModelError",
    "ModelNotFoundError",
    "PredictionError",
    "SignLanguageError",
]
