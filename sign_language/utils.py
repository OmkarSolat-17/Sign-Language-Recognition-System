"""Small, dependency-light helpers shared by training and prediction.

Everything in this module works without TensorFlow, OpenCV or MediaPipe
installed, which keeps it unit-testable in a minimal environment.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from typing import Any

from .exceptions import CameraError

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure root logging once, in a CLI-friendly format."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def expand_and_clip_bbox(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    frame_width: int,
    frame_height: int,
    padding: int = 0,
) -> tuple[int, int, int, int] | None:
    """Expand a bounding box by ``padding`` and clip it to the frame.

    Returns ``None`` when the resulting box is empty (hand partially or
    fully outside the frame, degenerate landmarks, ...), so callers can
    simply skip that detection instead of crashing on an empty crop —
    a bug the original script could hit on every frame edge.
    """
    if frame_width <= 0 or frame_height <= 0:
        return None
    x_min = max(0, int(x_min) - padding)
    y_min = max(0, int(y_min) - padding)
    x_max = min(int(frame_width), int(x_max) + padding)
    y_max = min(int(frame_height), int(y_max) + padding)
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def landmark_bbox(
    landmarks: Iterable[Any],
    frame_width: int,
    frame_height: int,
    padding: int = 0,
) -> tuple[int, int, int, int] | None:
    """Bounding box (pixels) around normalised ``(x, y)`` landmarks.

    Accepts MediaPipe landmark objects (``.x`` / ``.y`` attributes) as
    well as plain ``(x, y)`` tuples, which makes it testable without
    MediaPipe installed.
    """
    points: list[tuple[float, float]] = []
    for landmark in landmarks:
        if hasattr(landmark, "x") and hasattr(landmark, "y"):
            points.append((float(landmark.x), float(landmark.y)))
        else:
            x, y = landmark
            points.append((float(x), float(y)))
    if not points:
        return None
    xs = [p[0] * frame_width for p in points]
    ys = [p[1] * frame_height for p in points]
    return expand_and_clip_bbox(
        int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)),
        frame_width, frame_height, padding=padding,
    )


class PredictionSmoother:
    """Majority-vote smoothing over the last *N* predicted labels.

    Raw per-frame CNN predictions flicker.  Keeping a short history and
    displaying the most frequent recent label makes the on-screen text
    stable without hiding genuine sign changes for long.
    """

    def __init__(self, window_size: int = 5) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size!r}.")
        self.window_size = window_size
        self._history: deque[str] = deque(maxlen=window_size)

    def update(self, label: str) -> str:
        """Add a prediction and return the current smoothed label."""
        self._history.append(label)
        return Counter(self._history).most_common(1)[0][0]

    def reset(self) -> None:
        self._history.clear()

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)


def parse_video_source(source: str | int) -> int | str:
    """Parse a CLI video source: a camera index (``"0"``) or a file path.

    Raises :class:`~sign_language.exceptions.CameraError` for an empty
    value so the CLI fails fast with a helpful message.
    """
    if isinstance(source, int):
        if source < 0:
            raise CameraError(f"Camera index must be >= 0, got {source}.")
        return source
    text = str(source).strip()
    if not text:
        raise CameraError(
            "No video source given. Pass a camera index (e.g. --source 0) "
            "or a path to a video file."
        )
    if text.isdigit():
        return int(text)
    return text


def format_prediction(label: str, confidence: float, threshold: float) -> str:
    """Human-readable prediction text, marking low-confidence guesses."""
    suffix = "" if confidence >= threshold else " (?)"
    return f"{label} ({confidence:.0%}){suffix}"


def check_labels_match_model(labels: Sequence[str], output_units: int) -> None:
    """Guard against the classic labels/model mismatch.

    The original script re-derived class names from whatever happened to
    be in a hard-coded dataset folder at prediction time, so a renamed or
    reordered folder silently produced wrong labels.  Labels are now
    saved next to the model at training time and validated here.
    """
    from .exceptions import LabelsError

    if len(labels) != output_units:
        raise LabelsError(
            f"Label count ({len(labels)}) does not match the model's output "
            f"size ({output_units}). The labels file and the model were "
            "probably produced by different training runs — retrain the "
            "model or pass the matching --labels file."
        )
