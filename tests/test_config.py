"""Configuration validation: bad input must fail fast with a clear error."""

import pytest

from sign_language.config import PredictConfig, TrainConfig
from sign_language.exceptions import ConfigurationError


def test_train_config_defaults_derive_labels_path(tmp_path):
    config = TrainConfig(model_path=tmp_path / "m" / "model.keras")
    assert config.labels_path == tmp_path / "m" / "labels.json"
    assert config.img_size == 64


@pytest.mark.parametrize(
    "kwargs",
    [
        {"img_size": 0},
        {"img_size": 16},
        {"epochs": 0},
        {"batch_size": -1},
        {"test_size": 0.0},
        {"test_size": 1.0},
        {"test_size": 1.5},
        {"learning_rate": 0},
        {"patience": -1},
        {"model_path": "model.txt"},
    ],
)
def test_train_config_rejects_invalid_values(kwargs):
    with pytest.raises(ConfigurationError):
        TrainConfig(**kwargs)


def test_train_config_accepts_legacy_h5(tmp_path):
    config = TrainConfig(model_path=tmp_path / "model.h5")
    assert config.model_path.suffix == ".h5"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"bbox_padding": -1},
        {"camera_index": -1},
        {"img_size": 0},
    ],
)
def test_predict_config_rejects_invalid_values(kwargs):
    with pytest.raises(ConfigurationError):
        PredictConfig(**kwargs)


def test_predict_config_defaults():
    config = PredictConfig()
    assert config.confidence_threshold == 0.5
    assert config.labels_path.name == "labels.json"
