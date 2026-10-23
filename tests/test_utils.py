"""Pure helpers: bounding boxes, smoothing, sources, label checks."""

import pytest

from sign_language.exceptions import CameraError, LabelsError
from sign_language.utils import (
    PredictionSmoother,
    check_labels_match_model,
    expand_and_clip_bbox,
    format_prediction,
    landmark_bbox,
    parse_video_source,
)


def test_expand_and_clip_bbox_pads_and_clips():
    assert expand_and_clip_bbox(5, 5, 50, 50, 100, 100, padding=20) == (0, 0, 70, 70)
    assert expand_and_clip_bbox(90, 90, 100, 100, 100, 100, padding=20) == (70, 70, 100, 100)


def test_expand_and_clip_bbox_empty_returns_none():
    assert expand_and_clip_bbox(50, 50, 50, 60, 100, 100) is None
    assert expand_and_clip_bbox(0, 0, 10, 10, 0, 100) is None


def test_landmark_bbox_from_tuples():
    bbox = landmark_bbox([(0.1, 0.2), (0.5, 0.6), (0.3, 0.4)], 100, 200, padding=0)
    assert bbox == (10, 40, 50, 120)


def test_landmark_bbox_from_mediapipe_like_objects():
    class LM:
        def __init__(self, x, y):
            self.x, self.y = x, y

    bbox = landmark_bbox([LM(0.0, 0.0), LM(1.0, 1.0)], 640, 480, padding=10)
    assert bbox == (0, 0, 640, 480)


def test_landmark_bbox_empty():
    assert landmark_bbox([], 100, 100) is None


def test_smoother_majority_vote():
    smoother = PredictionSmoother(window_size=3)
    assert smoother.update("A") == "A"
    assert smoother.update("B") == "A"  # A,B -> tie broken by first-seen majority
    assert smoother.update("B") == "B"
    smoother.reset()
    assert smoother.history == ()
    assert smoother.update("C") == "C"


def test_smoother_rejects_bad_window():
    with pytest.raises(ValueError):
        PredictionSmoother(window_size=0)


def test_parse_video_source():
    assert parse_video_source("0") == 0
    assert parse_video_source(2) == 2
    assert parse_video_source("clip.mp4") == "clip.mp4"
    with pytest.raises(CameraError):
        parse_video_source("  ")
    with pytest.raises(CameraError):
        parse_video_source(-1)


def test_format_prediction_marks_low_confidence():
    assert format_prediction("A", 0.9, 0.5) == "A (90%)"
    assert format_prediction("A", 0.3, 0.5) == "A (30%) (?)"


def test_check_labels_match_model():
    check_labels_match_model(["A", "B"], 2)  # no error
    with pytest.raises(LabelsError, match="does not match"):
        check_labels_match_model(["A"], 2)
