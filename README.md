# Parkinson's Speech Classification with SpeechBrain

Refactored notebook into a reproducible SpeechBrain project for binary Parkinson's detection from short speech recordings.

## Purpose and Results

The goal of this project is to provide a reproducible pipeline for training and evaluating multiple SpeechBrain recipes on the Italian Parkinson Voice & Speech dataset, so model performance can be compared consistently under the same data splits.

Results are captured at the end of each run as metrics (validation error rate and accuracy by default) and stored in `results/<model>/<seed>/` with a summary table appended to `reports/results.md`. See those files for the latest numbers from your runs.

## Project Layout

- `data/raw/` – downloaded dataset (not committed)
- `data/manifests/` – JSON manifests built by `scripts/prepare_manifests.py`
- `recipes/parkinsons_binary/` – model recipes (xvector, ECAPA-TDNN, wav2vec2, WavLM, HuBERT)
- `src/parkinsons_speech/` – data prep and utility helpers
- `scripts/` – CLI for manifest prep, training automation, prediction stub
- `reports/` – results and figures placeholder

## Setup (Make)

Download and clone the repo, then run:

```
git clone <your-repo-url>
cd parkinson-detector

make install
```

## Data

1. Download the Italian Parkinson Voice & Speech dataset and place it under `data/raw/italian_parkinson`.
2. Build manifests (speaker-level split by default):

```
make download
make data
```

## Train

Run any recipe with:

```
make train MODEL=xvector
```

Swap `xvector` for `ecapa_tdnn`, `wav2vec2`, `wavlm`, or `hubert`. Checkpoints and logs go to `results/<model>/<seed>/`.

## One-command sweep

```
make all
```

This prepares manifests, trains each recipe, and appends a results table stub to `reports/results.md`.

## Results

See `reports/results.md` for your runs. Default metrics: validation error rate and accuracy (speaker-level split recommended).

## Notes

- Splitting supports `--split_by speaker` (default, no speaker overlap) or `--split_by file` (stratified).
- `scripts/predict.py` provides a simple checkpoint loader for single-file predictions.
