# Architecture

This project packages multiple SpeechBrain recipes behind a shared data-prep and training workflow for binary Parkinson's detection.

## Component diagram (text)
- **Data prep (scripts/prepare_manifests.py, src/parkinsons_speech/data_prep.py):** walks the raw dataset, infers labels/speakers, computes durations, and writes JSON manifests with a `{data_root}` placeholder.
- **Recipes (recipes/parkinsons_binary/*):** SpeechBrain experiment folders (train script + YAML) that consume manifests and emit checkpoints, logs, and metrics.
- **Automation (Makefile, scripts/run_all.sh):** one-command entry points to run manifest prep, individual training, or a sweep across all recipes.
- **Results/reporting (reports/):** human-readable tables of validation metrics produced after each run.
- **Prediction stub (scripts/predict.py):** loads a saved checkpoint to score a single WAV file for quick smoke checks.

## Data model overview
- **Record manifest fields:** `wav` (path with placeholder), `length` (seconds), `label` (`parkinson`/`not_parkinson`), `speaker` (folder-derived ID).
- **Splits:** speaker-level (default) uses stratified train/val/test partitions without speaker overlap; file-level stratifies individual examples.

## Key flows
1. **Manifest generation:** `scripts/prepare_manifests.py --data_root <path>` → scans WAVs → infers labels/speakers → splits data → writes `data/manifests/{train,valid,test}.json` and `split_summary.json`.
2. **Training a recipe:** `make train MODEL=xvector` → SpeechBrain train script loads manifests → trains on GPU/CPU → saves checkpoints + logs under `results/xvector/<seed>/`.
3. **Prediction:** `scripts/predict.py --hparams ... --checkpoint_dir ... --wav ...` → loads trained model → outputs predicted label/score for the supplied audio.

## Module boundaries and responsibilities
- `src/parkinsons_speech/data_prep.py`: dataset scanning, label inference, duration calculation, stratified splitting, manifest writing.
- `src/parkinsons_speech/utils.py`: reproducibility utilities (seeding, directory helpers), waveform cropping, label encoder prep.
- `src/parkinsons_speech/eval.py`: thin wrappers over scikit-learn metrics and reports.
- `scripts/*.py`: CLI wrappers that orchestrate the modules without adding training logic.
- `recipes/parkinsons_binary/*`: SpeechBrain-specific code + hyperparameters, isolated per model.

## Why these choices
- **SpeechBrain recipes** keep experimental configurations explicit and reproducible for portfolio reviewers.
- **Manifests with `{data_root}` placeholders** allow portable paths across machines (local, Colab, server) without editing YAML.
- **Make targets + scripts** provide a minimal but predictable developer experience without extra tooling.
