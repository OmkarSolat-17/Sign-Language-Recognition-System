"""Dataset discovery/scanning/loading and labels persistence."""

import numpy as np
import pytest

from sign_language.data import (
    discover_classes,
    load_dataset,
    load_labels,
    save_labels,
    scan_dataset,
)
from sign_language.exceptions import DatasetError, DatasetNotFoundError, LabelsError


def test_discover_classes_sorted(tiny_dataset):
    assert discover_classes(tiny_dataset) == ["A", "B"]


def test_discover_classes_missing_dir(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="SLR_DATASET_PATH"):
        discover_classes(tmp_path / "does-not-exist")


def test_discover_classes_file_not_dir(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(DatasetError, match="not a directory"):
        discover_classes(file_path)


def test_discover_classes_no_subfolders(tmp_path):
    (tmp_path / "img.jpg").write_bytes(b"x")
    with pytest.raises(DatasetError, match="No class sub-folders"):
        discover_classes(tmp_path)


def test_scan_counts_and_ignores_non_images(tiny_dataset):
    summary = scan_dataset(tiny_dataset)
    assert summary.classes == ["A", "B"]
    assert summary.counts == {"A": 3, "B": 2}  # .txt, stray and hidden files ignored
    assert summary.total_images == 5


def test_scan_empty_dataset_raises(tmp_path):
    (tmp_path / "A").mkdir()
    with pytest.raises(DatasetError, match="No images"):
        scan_dataset(tmp_path)


def test_load_dataset_decodes_and_skips_corrupt(real_image_dataset):
    pytest.importorskip("cv2")
    X, y, classes, report = load_dataset(real_image_dataset, img_size=16)
    assert classes == ["A", "B"]
    assert X.shape == (8, 16, 16, 3)
    assert X.dtype == np.float32
    assert float(X.max()) <= 1.0
    assert sorted(set(y.tolist())) == [0, 1]
    assert report.loaded == 8
    assert report.skipped_unreadable == 1  # corrupt.png skipped, not fatal


def test_load_dataset_rejects_bad_img_size(tiny_dataset):
    pytest.importorskip("cv2")
    with pytest.raises(DatasetError):
        load_dataset(tiny_dataset, img_size=0)


def test_load_dataset_all_corrupt_raises(tmp_path):
    pytest.importorskip("cv2")
    class_dir = tmp_path / "ds" / "A"
    class_dir.mkdir(parents=True)
    (class_dir / "bad.jpg").write_bytes(b"nope")
    (tmp_path / "ds" / "B").mkdir()
    (tmp_path / "ds" / "B" / "bad.jpg").write_bytes(b"nope")
    with pytest.raises(DatasetError, match="none of them could be decoded"):
        load_dataset(tmp_path / "ds")


def test_labels_round_trip(tmp_path):
    path = save_labels(["A", "B", "hello"], tmp_path / "labels.json")
    assert load_labels(path) == ["A", "B", "hello"]


@pytest.mark.parametrize("labels", [[], ["A", "A"], ["A", 3], "AB"])
def test_save_labels_rejects_invalid(tmp_path, labels):
    with pytest.raises(LabelsError):
        save_labels(labels, tmp_path / "labels.json")


def test_load_labels_missing_file(tmp_path):
    with pytest.raises(LabelsError, match="retrain"):
        load_labels(tmp_path / "labels.json")


def test_load_labels_invalid_json(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LabelsError, match="not valid JSON"):
        load_labels(path)


def test_load_labels_wrong_shape(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text('{"A": 0}', encoding="utf-8")
    with pytest.raises(LabelsError, match="non-empty JSON list"):
        load_labels(path)
