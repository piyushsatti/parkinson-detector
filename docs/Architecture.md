# Architecture

This project is a SpeechBrain-based pipeline for binary Parkinson's detection from short speech recordings. The code is organized to separate data preparation, model recipes, and reusable utilities.

## Top-level layout

- `data/` – local datasets and generated manifests (not committed)
- `recipes/parkinsons_binary/` – model recipes (xvector, ECAPA-TDNN, wav2vec2, WavLM, HuBERT)
- `scripts/` – CLI helpers for preparing manifests, training sweeps, and prediction
- `src/parkinsons_speech/` – shared utilities and data prep helpers
- `reports/` – results tables and figures
- `results/` – training outputs (checkpoints, logs, metrics)

## Data flow

1. **Raw data** lives under `data/raw/` (see `data/README_DATA.md`).
2. **Manifest generation** uses `scripts/prepare_manifests.py` to build JSON manifests in `data/manifests/`.
3. **Training** uses a recipe, e.g. `recipes/parkinsons_binary/xvector/train.py` plus its YAML hyperparameters. The `--data_folder` flag points to the raw data root.
4. **Outputs** are written to `results/<model>/<seed>/` and summarized in `reports/`.

## Key entry points

- Training: `recipes/parkinsons_binary/<recipe>/train.py`
- Hyperparameters: `recipes/parkinsons_binary/<recipe>/hparams/train.yaml`
- Manifests: `scripts/prepare_manifests.py`
- Prediction stub: `scripts/predict.py`

## Configuration conventions

- Manifests use `{data_root}` placeholders; pass `--data_folder` to training scripts.
- Splits support speaker-level (default) or file-level via `--split_by` in manifest prep.
