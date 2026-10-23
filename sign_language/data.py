"""Dataset discovery, validation and loading.

Expected on-disk layout (one sub-folder per sign)::

    dataset/
        A/
            img001.jpg
            ...
        B/
            ...
        hello/
            ...

Scanning (:func:`scan_dataset`) only looks at file names and never
decodes images, so it is fast and works without OpenCV installed.
Loading (:func:`load_dataset`) decodes images lazily, *skips and counts*
corrupt/unreadable files instead of crashing on the first bad one, and
raises a clear :class:`~sign_language.exceptions.DatasetError` when
nothing usable remains.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import DatasetError, DatasetNotFoundError, DependencyError, LabelsError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass
class DatasetSummary:
    """Result of scanning a dataset directory without decoding images."""

    root: Path
    classes: list[str]
    counts: dict[str, int] = field(default_factory=dict)
    image_paths: dict[str, list[Path]] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return sum(self.counts.values())

    @property
    def num_classes(self) -> int:
        return len(self.classes)


@dataclass
class LoadReport:
    """Statistics about a :func:`load_dataset` run."""

    loaded: int = 0
    skipped_unreadable: int = 0
    skipped_paths: list[Path] = field(default_factory=list)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_classes(dataset_path: str | Path) -> list[str]:
    """Return the sorted class names (sub-folder names) of a dataset.

    Raises:
        DatasetNotFoundError: the path does not exist.
        DatasetError: the path is not a directory, or contains no
            class sub-folders.
    """
    root = Path(dataset_path).expanduser()
    if not root.exists():
        raise DatasetNotFoundError(
            f"Dataset directory not found: '{root}'. "
            "Pass --dataset <path>, set the SLR_DATASET_PATH environment "
            "variable, or create a 'dataset/' folder with one sub-folder "
            "per sign (see docs/dataset.md)."
        )
    if not root.is_dir():
        raise DatasetError(f"Dataset path is not a directory: '{root}'.")

    classes = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    if not classes:
        raise DatasetError(
            f"No class sub-folders found in '{root}'. Expected a layout like "
            "dataset/A/*.jpg, dataset/B/*.jpg, ... (see docs/dataset.md)."
        )
    return classes


def scan_dataset(dataset_path: str | Path) -> DatasetSummary:
    """Scan a dataset: class names, per-class image counts and paths.

    Files at the dataset root, hidden files/folders and unsupported
    extensions are ignored (with a debug/warning log), never fatal.
    """
    root = Path(dataset_path).expanduser()
    classes = discover_classes(root)

    summary = DatasetSummary(root=root, classes=classes)
    for class_name in classes:
        class_dir = root / class_name
        paths = sorted(
            p for p in class_dir.iterdir()
            if _is_image_file(p) and not p.name.startswith(".")
        )
        summary.image_paths[class_name] = paths
        summary.counts[class_name] = len(paths)
        if not paths:
            logger.warning("Class '%s' contains no usable images.", class_name)

    stray = [p for p in root.iterdir() if _is_image_file(p)]
    if stray:
        logger.warning(
            "Ignoring %d image(s) placed directly in '%s' — images must live "
            "inside a class sub-folder.", len(stray), root,
        )
    if summary.total_images == 0:
        raise DatasetError(
            f"No images found in '{root}'. Supported extensions: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    return summary


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "OpenCV is required to decode images but is not installed. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc
    return cv2


def load_dataset(
    dataset_path: str | Path,
    img_size: int = 64,
) -> tuple[object, object, list[str], LoadReport]:
    """Load all images into memory as normalised float arrays.

    Returns ``(X, y, classes, report)`` where ``X`` has shape
    ``(n, img_size, img_size, 3)`` with values in ``[0, 1]`` and ``y``
    holds integer class indices matching ``classes``.

    Corrupt, unreadable or wrongly-shaped images are skipped and counted
    in the report instead of aborting the whole run (the original script
    crashed inside ``cv2.resize`` on the first such file).
    """
    if img_size <= 0:
        raise DatasetError(f"img_size must be positive, got {img_size!r}.")
    cv2 = _require_cv2()
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "NumPy is required but is not installed. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc

    summary = scan_dataset(dataset_path)
    report = LoadReport()
    images: list = []
    labels: list[int] = []

    for label_index, class_name in enumerate(summary.classes):
        for img_path in summary.image_paths[class_name]:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    raise ValueError("cv2.imread returned None (corrupt or unsupported file)")
                image = cv2.resize(image, (img_size, img_size))
            except Exception as exc:  # one bad file must not kill training
                logger.warning("Skipping unreadable image %s (%s)", img_path, exc)
                report.skipped_unreadable += 1
                report.skipped_paths.append(img_path)
                continue
            images.append(image)
            labels.append(label_index)
            report.loaded += 1

    if report.loaded == 0:
        raise DatasetError(
            f"Found {summary.total_images} image file(s) in '{summary.root}' "
            "but none of them could be decoded. Check that the files are "
            "real images and that OpenCV supports their format."
        )
    for label_index, class_name in enumerate(summary.classes):
        if labels.count(label_index) == 0:
            logger.warning(
                "Class '%s' had %d file(s) but none could be decoded; it "
                "will be missing from the trained model.",
                class_name, summary.counts[class_name],
            )

    X = np.asarray(images, dtype=np.float32) / 255.0
    y = np.asarray(labels, dtype=np.int64)
    logger.info(
        "Loaded %d image(s) across %d class(es); skipped %d unreadable file(s).",
        report.loaded, summary.num_classes, report.skipped_unreadable,
    )
    return X, y, list(summary.classes), report


def split_dataset(X, y, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split with friendly validation errors.

    Stratification keeps every class represented in both splits, but it
    needs at least 2 samples per class — that precondition is checked
    here so users get a readable error instead of a raw scikit-learn
    traceback.
    """
    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "scikit-learn and NumPy are required for the train/test split "
            "but are not installed. Install them with: "
            "pip install -r requirements.txt"
        ) from exc

    y_array = np.asarray(y)
    if y_array.size == 0:
        raise DatasetError("Cannot split an empty dataset.")
    classes, counts = np.unique(y_array, return_counts=True)
    too_small = [int(c) for c, n in zip(classes, counts, strict=True) if n < 2]
    if too_small:
        raise DatasetError(
            "Each class needs at least 2 images for a stratified train/test "
            f"split; class index/indices {too_small} have fewer. Add more "
            "images for those signs or reduce --test-size."
        )
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y_array, shuffle=True,
    )


def save_labels(labels: list[str], labels_path: str | Path) -> Path:
    """Persist class labels as JSON next to the trained model."""
    if isinstance(labels, (str, bytes)) or not isinstance(labels, (list, tuple)):
        raise LabelsError(f"Labels must be a non-empty list of strings, got {labels!r}.")
    labels = list(labels)
    if not labels or not all(isinstance(item, str) and item for item in labels):
        raise LabelsError(f"Labels must be a non-empty list of strings, got {labels!r}.")
    if len(set(labels)) != len(labels):
        raise LabelsError(f"Labels contain duplicates: {labels!r}.")
    path = Path(labels_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %d label(s) to %s", len(labels), path)
    return path


def load_labels(labels_path: str | Path) -> list[str]:
    """Load and validate class labels written by :func:`save_labels`."""
    path = Path(labels_path).expanduser()
    if not path.is_file():
        raise LabelsError(
            f"Labels file not found: '{path}'. It is written automatically "
            "by train.py next to the model; pass --labels <path> if yours "
            "lives elsewhere, or retrain the model."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabelsError(f"Labels file '{path}' is not valid JSON: {exc}.") from exc
    if (
        not isinstance(data, list)
        or not data
        or not all(isinstance(item, str) and item for item in data)
    ):
        raise LabelsError(
            f"Labels file '{path}' must contain a non-empty JSON list of "
            f"class-name strings, got: {data!r}."
        )
    if len(set(data)) != len(data):
        raise LabelsError(f"Labels file '{path}' contains duplicate labels: {data!r}.")
    return list(data)
