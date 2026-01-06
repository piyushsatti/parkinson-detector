# Parkinson's Speech Classification with SpeechBrain

Refactored notebook into a reproducible SpeechBrain project for binary Parkinson's detection from short speech recordings.

## Project Layout

- `data/raw/` – downloaded dataset (not committed)
- `data/manifests/` – JSON manifests built by `scripts/prepare_manifests.py`
- `recipes/parkinsons_binary/` – model recipes (xvector, ECAPA-TDNN, wav2vec2, WavLM, HuBERT)
- `src/parkinsons_speech/` – data prep and utility helpers
- `scripts/` – CLI for manifest prep, training automation, prediction stub
- `reports/` – results and figures placeholder

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src:$PYTHONPATH
```

## Data

1. Download the Italian Parkinson Voice & Speech dataset (see `data/README_DATA.md`).
2. Build manifests (speaker-level split by default):

```
python scripts/prepare_manifests.py --data_root data/raw/italian_parkinson --out_dir data/manifests --split_by speaker
```

## Train

Run any recipe with:

```
python recipes/parkinsons_binary/xvector/train.py recipes/parkinsons_binary/xvector/hparams/train.yaml --data_folder data/raw/italian_parkinson
```

Swap `xvector` for `ecapa_tdnn`, `wav2vec2`, `wavlm`, or `hubert`. Checkpoints and logs go to `results/<model>/<seed>/`.

## One-command sweep

```
bash scripts/run_all.sh data/raw/italian_parkinson
```

This prepares manifests, trains each recipe, and appends a results table stub to `reports/results.md`.

## Results

See `reports/results.md` for your runs. Default metrics: validation error rate and accuracy (speaker-level split recommended).

## Notes

- Manifest paths use `{data_root}` placeholders; pass `--data_folder` to training scripts.
- Splitting supports `--split_by speaker` (default, no speaker overlap) or `--split_by file` (stratified).
- `scripts/predict.py` provides a simple checkpoint loader for single-file predictions.
