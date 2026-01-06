# Parkinson's Speech Classification with SpeechBrain

Refactored SpeechBrain pipeline for binary Parkinson's detection from short speech recordings using multiple recipe baselines.

## Demo
- _No hosted demo yet._ Training produces results tables under `reports/results.md` with per-run metrics.

## Features
- Reproducible SpeechBrain recipes for xvector, ECAPA-TDNN, wav2vec2, WavLM, and HuBERT under a unified `recipes/parkinsons_binary` layout.
- Automated manifest generation with speaker-level or file-level splits via `scripts/prepare_manifests.py` and the `make data` target.
- One-command sweep (`make all`) that prepares data, runs every recipe, and appends a summary stub to `reports/results.md`.
- Simple prediction helper (`scripts/predict.py`) to load a trained checkpoint and score a single WAV file.

## Tech Stack
- Python, Poetry
- SpeechBrain for training recipes
- Torchaudio and scikit-learn for audio metadata and evaluation

## Architecture Overview
See `docs/Architecture.md` for a diagram and data flow between manifest prep, recipes, and outputs.

## Local Setup
```bash
git clone <your-repo-url>
cd parkinson-detector

# Install dependencies
make install

# (Optional) download the dataset archive to data/raw/italian_parkinson
make download

# Build manifests (speaker-level split by default)
make data
```

## Usage Examples
- Train a single recipe: `make train MODEL=xvector`
- Switch recipe: `make train MODEL=ecapa_tdnn` (or `wav2vec2`, `wavlm`, `hubert`)
- Predict on one WAV: `make predict WAV=path/to/audio.wav CKPT=results/xvector/1234/HPARAMS HP=recipes/parkinsons_binary/xvector/hparams/train.yaml`
- Run all recipes with manifests: `make all`

## Project Structure
```
.
├── Makefile                    # Common automation targets
├── recipes/parkinsons_binary/  # SpeechBrain recipes (code + hparams per model)
├── scripts/                    # CLI helpers for manifests, training sweeps, prediction
├── src/parkinsons_speech/      # Data prep, evaluation, and utility helpers
├── data/                       # Raw data + generated manifests (not committed)
├── reports/                    # Results tables and figures
└── docs/                       # Architecture + Colab usage guides
```

## Known Limitations
- No automated tests or CI are configured; run commands locally to validate changes.
- Dataset is not bundled—see `make download` or place the Italian Parkinson Voice & Speech data under `data/raw/italian_parkinson`.
- Current metrics and checkpoints are local only; `reports/results.md` captures outputs from your own runs.

## Roadmap
- Add lightweight evaluation scripts for quick health checks.
- Provide example checkpoints and manifest samples for faster onboarding.

## License
MIT
