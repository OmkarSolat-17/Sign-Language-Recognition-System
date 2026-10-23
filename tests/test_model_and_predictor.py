"""Model building/loading and the SignPredictor wrapper.

TensorFlow tests skip cleanly when TensorFlow is not installed, so the
suite stays fast in the lightweight CI job.
"""

import numpy as np
import pytest

from sign_language.exceptions import LabelsError, ModelError, ModelNotFoundError
from sign_language.model import build_model, load_model


def test_build_model_rejects_too_few_classes():
    with pytest.raises(ModelError, match="at least 2 classes"):
        build_model(num_classes=1)


def test_build_model_rejects_bad_img_size():
    with pytest.raises(ModelError):
        build_model(num_classes=3, img_size=0)


def test_build_model_rejects_too_small_img_size():
    # 16x16 shrinks to a zero-sized feature map after 3 conv/pool blocks;
    # this must be a friendly ModelError, not a deep Keras ValueError.
    with pytest.raises(ModelError, match="at least 32"):
        build_model(num_classes=3, img_size=16)


def test_load_model_missing_file(tmp_path):
    with pytest.raises(ModelNotFoundError, match="Train a model first"):
        load_model(tmp_path / "model.keras")


def test_build_and_round_trip_model(tmp_path):
    pytest.importorskip("tensorflow")
    from sign_language.data import save_labels
    from sign_language.model import save_model
    from sign_language.predictor import SignPredictor

    model = build_model(num_classes=3, img_size=32)
    assert model.output_shape == (None, 3)
    model_path = save_model(model, tmp_path / "model.keras")
    save_labels(["A", "B", "C"], tmp_path / "labels.json")

    predictor = SignPredictor(model_path, img_size=32)
    prediction = predictor.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert prediction.label in {"A", "B", "C"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.probabilities) == 3


def test_predictor_rejects_label_model_mismatch(tmp_path):
    pytest.importorskip("tensorflow")
    from sign_language.data import save_labels
    from sign_language.model import save_model
    from sign_language.predictor import SignPredictor

    save_model(build_model(num_classes=2, img_size=32), tmp_path / "model.keras")
    save_labels(["A", "B", "C"], tmp_path / "labels.json")  # 3 labels, 2 outputs
    with pytest.raises(LabelsError, match="does not match"):
        SignPredictor(tmp_path / "model.keras", img_size=16)


def test_preprocess_rejects_wrong_shape():
    from sign_language.predictor import preprocess_image

    with pytest.raises(Exception, match="colour image"):
        preprocess_image(np.zeros((10, 10), dtype=np.uint8))


def test_preprocess_normalises_and_batches():
    from sign_language.predictor import preprocess_image

    batch = preprocess_image(np.full((16, 16, 3), 255, dtype=np.uint8), img_size=16)
    assert batch.shape == (1, 16, 16, 3)
    assert float(batch.max()) == pytest.approx(1.0)


def test_split_dataset_requires_two_per_class(real_image_dataset):
    pytest.importorskip("sklearn")
    from sign_language.data import load_dataset, split_dataset

    X, y, _classes, _report = load_dataset(real_image_dataset, img_size=8)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.25)
    assert len(X_train) + len(X_test) == len(X)
    assert set(y_train.tolist()) == {0, 1}
    assert set(y_test.tolist()) == {0, 1}
