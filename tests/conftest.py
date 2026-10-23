"""Shared pytest fixtures: tiny on-disk datasets built in tmp dirs."""

from __future__ import annotations

import pytest


@pytest.fixture
def tiny_dataset(tmp_path):
    """dataset/{A,B}/ with a few (undecodable but correctly-named) files.

    Scanning never decodes images, so empty files are enough to test
    discovery, counting and validation.  Tests that *decode* images use
    OpenCV to write real ones and skip when OpenCV is absent.
    """
    root = tmp_path / "dataset"
    for class_name, count in {"A": 3, "B": 2}.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            (class_dir / f"img{i}.jpg").write_bytes(b"not-a-real-jpeg")
    (root / "B" / "notes.txt").write_text("ignored", encoding="utf-8")
    (root / "stray.jpg").write_bytes(b"ignored: not inside a class folder")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "x.jpg").write_bytes(b"ignored")
    return root


@pytest.fixture
def real_image_dataset(tmp_path):
    """A dataset of real, decodable images (requires OpenCV + NumPy)."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    root = tmp_path / "dataset"
    for label, class_name in enumerate(["A", "B"]):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(4):
            image = np.full((20, 20, 3), 40 * label + i, dtype=np.uint8)
            assert cv2.imwrite(str(class_dir / f"img{i}.png"), image)
    # One corrupt file that must be skipped, not fatal.
    (root / "A" / "corrupt.png").write_bytes(b"this is not a png")
    return root
