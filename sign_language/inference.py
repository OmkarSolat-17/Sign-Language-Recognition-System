"""Real-time (webcam/video) and single-image prediction, plus its CLI.

Modes:

* **webcam** (default) — MediaPipe finds the hand, the CNN classifies
  the cropped hand region, the label is drawn on the frame.  Press
  ``q`` or ``Esc`` to quit.
* **video file** — ``--source clip.mp4``; same pipeline, file instead
  of a camera.
* **single image** — ``--image photo.jpg``; classifies the whole image
  (or its largest hand crop when MediaPipe is available), prints the
  result and optionally writes an annotated copy with ``--output``.
  This mode needs no camera and is what the smoke tests / CI use.

All resources (camera, windows, MediaPipe graph) are released in
``finally`` blocks, consecutive frame-read failures abort with a clear
error instead of spinning forever, and every expected failure raises a
:class:`~sign_language.exceptions.SignLanguageError` subclass.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import PredictConfig
from .exceptions import CameraError, DependencyError, PredictionError, SignLanguageError
from .predictor import SignPredictor
from .utils import (
    PredictionSmoother,
    format_prediction,
    landmark_bbox,
    parse_video_source,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "OpenCV is required for camera/image input but is not installed. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc
    return cv2


def _load_mediapipe_hands():
    """Return ``(hands_solution, drawing_utils, HAND_CONNECTIONS)`` or raise."""
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DependencyError(
            "MediaPipe is required for hand detection but is not installed. "
            "Install it with: pip install -r requirements.txt — or run "
            "with --no-mediapipe to classify the whole frame instead."
        ) from exc
    try:
        mp_hands = mp.solutions.hands
        return mp_hands, mp.solutions.drawing_utils, mp_hands.HAND_CONNECTIONS
    except AttributeError as exc:  # pragma: no cover - newer MediaPipe versions
        raise DependencyError(
            "This MediaPipe version does not expose 'mp.solutions.hands'. "
            "Install a compatible version with: pip install -r requirements.txt"
        ) from exc


# ---------------------------------------------------------------------------
# Single-image mode
# ---------------------------------------------------------------------------

def run_image(predictor: SignPredictor, image_path: str | Path,
              output_path: str | Path | None = None) -> str:
    """Classify one image file; returns the formatted prediction text."""
    cv2 = _require_cv2()
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise PredictionError(
            f"Image file not found: '{path}'. Check the --image path and try again."
        )
    frame = cv2.imread(str(path))
    if frame is None:
        raise PredictionError(
            f"Could not decode image '{path}'. It may be corrupt or in a "
            "format OpenCV does not support (try .jpg or .png)."
        )
    prediction = predictor.predict(frame)
    text = format_prediction(prediction.label, prediction.confidence, predictor.confidence_threshold)
    if output_path is not None:
        out = Path(output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if not cv2.imwrite(str(out), frame):
            raise PredictionError(f"Failed to write annotated image to '{out}'.")
        logger.info("Wrote annotated image to %s", out)
    return text


# ---------------------------------------------------------------------------
# Webcam / video mode
# ---------------------------------------------------------------------------

def run_stream(predictor: SignPredictor, config: PredictConfig,
               source: int | str = 0, use_mediapipe: bool = True,
               show_window: bool = True) -> None:
    """Run the live prediction loop until ``q``/``Esc`` or end of stream."""
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        kind = "camera" if isinstance(source, int) else "video file"
        raise CameraError(
            f"Could not open {kind} source '{source}'. "
            + ("Check that the camera is connected, not used by another app, "
               "and that you granted camera permission."
               if isinstance(source, int) else
               "Check that the file exists and is a supported video format.")
        )

    hands = None
    mp_draw = hand_connections = None
    if use_mediapipe:
        mp_hands, mp_draw_module, hand_connections = _load_mediapipe_hands()
        hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        mp_draw = mp_draw_module

    smoother = PredictionSmoother(window_size=config.smoothing_window)
    frame_failures = 0
    logger.info("Starting prediction — press 'q' or Esc to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                frame_failures += 1
                if isinstance(source, str) and frame_failures > 0 and not capture.isOpened():
                    break  # end of video file
                if frame_failures >= config.max_frame_failures:
                    raise CameraError(
                        f"Failed to read {frame_failures} consecutive frames "
                        f"from source '{source}'. The camera may have been "
                        "disconnected or the video file may be corrupt."
                    )
                if not isinstance(source, int):
                    break  # video file ended normally
                continue
            frame_failures = 0
            height, width = frame.shape[:2]

            regions: list[tuple] = []  # (bbox or None-for-whole-frame, landmarks or None)
            if hands is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        bbox = landmark_bbox(
                            hand_landmarks.landmark, width, height,
                            padding=config.bbox_padding,
                        )
                        if bbox is not None:
                            regions.append((bbox, hand_landmarks))
                # No hand found this frame: regions stays empty, frame is shown as-is.
            else:
                regions.append((None, None))

            for bbox, hand_landmarks in regions:
                if bbox is None:
                    crop, origin_x, origin_y = frame, 10, 30
                else:
                    x_min, y_min, x_max, y_max = bbox
                    crop = frame[y_min:y_max, x_min:x_max]
                    origin_x, origin_y = x_min, max(y_min - 10, 20)
                if crop is None or crop.size == 0:
                    continue
                try:
                    prediction = predictor.predict(crop)
                except SignLanguageError as exc:
                    logger.warning("Skipping frame prediction: %s", exc)
                    continue
                smoothed = smoother.update(prediction.label)
                text = format_prediction(
                    smoothed, prediction.confidence, predictor.confidence_threshold
                )
                if bbox is not None:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, text, (origin_x, origin_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if hand_landmarks is not None and mp_draw is not None:
                    mp_draw.draw_landmarks(frame, hand_landmarks, hand_connections)

            if show_window:
                cv2.imshow("Sign Language Recognition (q/Esc to quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    logger.info("Quit requested by user.")
                    break
    finally:
        capture.release()
        if hands is not None:
            hands.close()
        if show_window:
            cv2.destroyAllWindows()
    logger.info("Prediction stopped; all resources released.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Recognise sign language from a webcam, a video file, or a single image.",
        epilog="Examples:  python predict.py   |   python predict.py --image photo.jpg   |   "
               "python predict.py --source clip.mp4",
    )
    parser.add_argument("--model", default=None,
                        help="Model file (default: $SLR_MODEL_PATH or ./model.keras).")
    parser.add_argument("--labels", default=None,
                        help="Labels JSON (default: labels.json next to the model).")
    parser.add_argument("--image", default=None, help="Classify a single image file and exit.")
    parser.add_argument("--output", default=None,
                        help="With --image: also save an annotated copy to this path.")
    parser.add_argument("--source", default=None,
                        help="Camera index (default 0) or path to a video file.")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="Camera index (deprecated alias; prefer --source).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence below which a prediction is marked '(?)' (default: 0.5).")
    parser.add_argument("--padding", type=int, default=20,
                        help="Pixels of padding around the detected hand box (default: 20).")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Model input size the model was trained with (default: 64).")
    parser.add_argument("--no-mediapipe", action="store_true",
                        help="Skip hand detection and classify the whole frame.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def config_from_args(args: argparse.Namespace) -> PredictConfig:
    kwargs: dict = {
        "confidence_threshold": args.threshold,
        "bbox_padding": args.padding,
        "img_size": args.img_size,
    }
    if args.model:
        kwargs["model_path"] = Path(args.model)
    if args.labels:
        kwargs["labels_path"] = Path(args.labels)
    if args.camera_index is not None:
        kwargs["camera_index"] = args.camera_index
    return PredictConfig(**kwargs)


def resolve_source(args: argparse.Namespace, config: PredictConfig) -> int | str:
    if args.source is not None:
        return parse_video_source(args.source)
    return config.camera_index


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(verbose=args.verbose)
    try:
        config = config_from_args(args)
        predictor = SignPredictor(
            config.model_path, config.labels_path,
            img_size=config.img_size,
            confidence_threshold=config.confidence_threshold,
        )
        if args.image:
            text = run_image(predictor, args.image, args.output)
            print(text)
        else:
            run_stream(
                predictor, config,
                source=resolve_source(args, config),
                use_mediapipe=not args.no_mediapipe,
            )
    except SignLanguageError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
