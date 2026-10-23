"""Custom exception hierarchy for the Sign Language Recognition System.

Every error that a user can reasonably trigger (missing dataset, corrupt
images, missing model, unavailable camera, ...) is raised as a subclass of
:class:`SignLanguageError` with an actionable message.  The command-line
entry points catch these, print a clean ``Error: ...`` message and exit
with a non-zero status code instead of dumping a raw traceback.
"""

from __future__ import annotations


class SignLanguageError(Exception):
    """Base class for all expected, user-facing errors in this project."""


class ConfigurationError(SignLanguageError):
    """Raised when a configuration value is invalid."""


class DependencyError(SignLanguageError):
    """Raised when an optional/heavy dependency is not installed.

    The message always includes the ``pip install`` command that fixes it.
    """


class DatasetError(SignLanguageError):
    """Raised when the dataset is missing, malformed or unusable."""


class DatasetNotFoundError(DatasetError):
    """Raised when the dataset directory does not exist."""


class ModelError(SignLanguageError):
    """Raised when a model cannot be built, saved or loaded."""


class ModelNotFoundError(ModelError):
    """Raised when the model file does not exist."""


class LabelsError(ModelError):
    """Raised when the class-labels file is missing or malformed."""


class CameraError(SignLanguageError):
    """Raised when the camera / video source cannot be opened or read."""


class PredictionError(SignLanguageError):
    """Raised when a single prediction cannot be produced."""
