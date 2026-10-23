"""CLI behaviour: --help works without heavy deps, errors exit cleanly.

These tests deliberately run *without* TensorFlow/OpenCV/MediaPipe
installed (they're only present in the full environment), which is
exactly the situation a new contributor hits first.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )


def test_train_help():
    result = run_cli("train.py", "--help")
    assert result.returncode == 0
    assert "--dataset" in result.stdout


def test_predict_help():
    result = run_cli("predict.py", "--help")
    assert result.returncode == 0
    assert "--image" in result.stdout


def test_train_missing_dataset_exits_1_with_clean_error(tmp_path):
    result = run_cli("train.py", "--dataset", str(tmp_path / "nope"))
    assert result.returncode == 1
    assert "Dataset directory not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_predict_missing_model_exits_1_with_clean_error(tmp_path):
    result = run_cli("predict.py", "--model", str(tmp_path / "model.keras"),
                     "--image", "whatever.jpg")
    assert result.returncode == 1
    assert "Labels file not found" in result.stderr or "Model file not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_train_rejects_invalid_epochs():
    result = run_cli("train.py", "--epochs", "0")
    assert result.returncode == 1
    assert "Traceback" not in result.stderr


def test_check_dataset_script(tmp_path):
    dataset = tmp_path / "dataset"
    for name in ("A", "B"):
        (dataset / name).mkdir(parents=True)
        (dataset / name / "a.jpg").write_bytes(b"x")
        (dataset / name / "b.jpg").write_bytes(b"x")
    result = run_cli("scripts/check_dataset.py", "--dataset", str(dataset))
    assert result.returncode == 0
    assert "trainable" in result.stdout
