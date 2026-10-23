"""Training pipeline and its command-line interface.

``python train.py`` (the thin wrapper at the repository root) and
``python -m sign_language.training`` both end up here.  Heavy imports
(TensorFlow, OpenCV, scikit-learn) happen inside functions so that
``--help`` and argument validation work everywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import TrainConfig
from .data import load_dataset, save_labels, scan_dataset, split_dataset
from .exceptions import SignLanguageError
from .model import build_model, save_model
from .utils import setup_logging

logger = logging.getLogger(__name__)


def run_training(config: TrainConfig) -> dict:
    """Train, evaluate and save a model. Returns a summary dict.

    Raises subclasses of :class:`~sign_language.exceptions.SignLanguageError`
    for every expected failure (missing dataset, too few images, missing
    dependencies, unwritable output path, ...).
    """
    summary = scan_dataset(config.dataset_path)
    logger.info(
        "Dataset '%s': %d class(es), %d image(s) — %s",
        summary.root, summary.num_classes, summary.total_images,
        ", ".join(f"{name}={summary.counts[name]}" for name in summary.classes),
    )
    if summary.num_classes < 2:
        from .exceptions import DatasetError

        raise DatasetError(
            "At least 2 classes are required to train a classifier, found "
            f"{summary.num_classes} in '{summary.root}'."
        )

    X, y, classes, report = load_dataset(config.dataset_path, config.img_size)
    if report.skipped_unreadable:
        logger.warning(
            "Skipped %d unreadable image(s) during loading.", report.skipped_unreadable
        )

    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    logger.info("Split: %d train / %d test image(s).", len(X_train), len(X_test))

    model = build_model(
        num_classes=len(classes),
        img_size=config.img_size,
        learning_rate=config.learning_rate,
        augment=config.augment,
    )
    model.summary(print_fn=logger.info)

    callbacks = _build_callbacks(config)
    history = model.fit(
        X_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test accuracy: %.2f%% (loss %.4f)", test_accuracy * 100, test_loss)

    save_model(model, config.model_path)
    save_labels(classes, config.labels_path)
    _save_history(history, config.model_path.with_name("training_history.json"))

    return {
        "classes": classes,
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "model_path": str(config.model_path),
        "labels_path": str(config.labels_path),
        "skipped_unreadable": report.skipped_unreadable,
    }


def _build_callbacks(config: TrainConfig) -> list:
    """Early stopping (+ optional augmentation note) — TF imported lazily."""
    from .model import _require_tensorflow

    tf = _require_tensorflow()
    callbacks = []
    if config.patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=config.patience,
                restore_best_weights=True,
            )
        )
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3))
    return callbacks


def _save_history(history, path: Path) -> None:
    try:
        serialisable = {
            key: [float(v) for v in values] for key, values in history.history.items()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serialisable, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved training history to %s", path)
    except Exception as exc:  # history is diagnostic only — never fail training over it
        logger.warning("Could not save training history to %s: %s", path, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train the sign-language CNN on a folder-per-class image dataset.",
        epilog="Example: python train.py --dataset dataset --epochs 20 --augment",
    )
    parser.add_argument("--dataset", default=None,
                        help="Dataset folder (default: $SLR_DATASET_PATH or ./dataset).")
    parser.add_argument("--model-output", default=None,
                        help="Where to save the model (default: $SLR_MODEL_PATH or ./model.keras).")
    parser.add_argument("--labels-output", default=None,
                        help="Where to save labels.json (default: next to the model).")
    parser.add_argument("--img-size", type=int, default=64, help="Square image size (default: 64).")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for testing, 0-1 (default: 0.2).")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--random-state", type=int, default=42, help="Split/train random seed.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience in epochs; 0 disables it (default: 5).")
    parser.add_argument("--augment", action="store_true",
                        help="Add random flip/rotation/zoom augmentation layers (training only).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    kwargs: dict = {
        "img_size": args.img_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "augment": args.augment,
    }
    if args.dataset:
        kwargs["dataset_path"] = Path(args.dataset)
    if args.model_output:
        kwargs["model_path"] = Path(args.model_output)
    if args.labels_output:
        kwargs["labels_path"] = Path(args.labels_output)
    return TrainConfig(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(verbose=args.verbose)
    try:
        config = config_from_args(args)
        result = run_training(config)
    except SignLanguageError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Training interrupted by user.")
        return 130
    logger.info(
        "Done. Model: %s | Labels: %s | Test accuracy: %.2f%%",
        result["model_path"], result["labels_path"], result["test_accuracy"] * 100,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
