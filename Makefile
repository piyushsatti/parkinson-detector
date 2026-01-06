SHELL := /bin/bash
PYTHON := poetry run python
DATA_ROOT ?= data/raw/italian_parkinson
MANIFEST_DIR ?= data/manifests
MODEL ?= xvector
DEVICE ?= cpu

.PHONY: help install data download train all predict clean

help:
	@echo "Targets:"
	@echo "  install    Install deps with Poetry"
	@echo "  data       Download dataset (if missing) and prepare manifests"
	@echo "  download   Download dataset archive and extract"
	@echo "  train      Train single model (MODEL=...)"
	@echo "  all        Run full sweep (all models)"
	@echo "  predict    Predict on a wav (WAV=... CKPT=... HP=...)"
	@echo "  clean      Remove training artifacts"

install:
	poetry install

download:
	bash scripts/download_dataset.sh $(DATA_ROOT)

data:
	$(PYTHON) scripts/prepare_manifests.py --data_root $(DATA_ROOT) --out_dir $(MANIFEST_DIR) --split_by speaker

train:
	$(PYTHON) recipes/parkinsons_binary/$(MODEL)/train.py recipes/parkinsons_binary/$(MODEL)/hparams/train.yaml --data_folder $(DATA_ROOT) --device $(DEVICE)

all:
	bash scripts/run_all.sh $(DATA_ROOT)

predict:
	$(PYTHON) scripts/predict.py --hparams $(HP) --checkpoint_dir $(CKPT) --data_folder $(DATA_ROOT) --wav $(WAV)

clean:
	rm -rf results
