.PHONY: help install install-dev test lint compile check-dataset train predict clean

PYTHON ?= python3
DATASET ?= dataset
MODEL ?= model.keras

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## Install runtime (ML) dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev:  ## Install lightweight dev dependencies (tests + lint)
	$(PYTHON) -m pip install -r requirements-dev.txt

test:  ## Run the test suite
	$(PYTHON) -m pytest

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check sign_language train.py predict.py scripts tests

compile:  ## Byte-compile all Python sources (syntax check)
	$(PYTHON) -m compileall -q sign_language train.py predict.py scripts tests

check-dataset:  ## Validate the dataset folder (DATASET=...)
	$(PYTHON) scripts/check_dataset.py --dataset $(DATASET)

train:  ## Train a model (DATASET=..., plus ARGS="--epochs 20" etc.)
	$(PYTHON) train.py --dataset $(DATASET) $(ARGS)

predict:  ## Run webcam prediction (MODEL=...)
	$(PYTHON) predict.py --model $(MODEL)

clean:  ## Remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f model.keras model.h5 labels.json training_history.json
