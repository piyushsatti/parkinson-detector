# Google Colab Guide

This guide shows how to run the project on Google Colab with a GPU runtime.

## 1) Start a GPU runtime

- In Colab: `Runtime` -> `Change runtime type` -> select `GPU`.

## 2) Clone the repo and install dependencies

```bash
!git clone <your-repo-url>
%cd parkinson-detector

!pip -q install poetry
!poetry install
```

Make sure the project package is importable:

```python
import os
os.environ["PYTHONPATH"] = "src:" + os.environ.get("PYTHONPATH", "")
```

## 3) Mount your dataset (recommended: Google Drive)

```python
from google.colab import drive
drive.mount("/content/drive")

DATA_ROOT = "/content/drive/MyDrive/italian_parkinson"
```

Place the dataset under `DATA_ROOT` so it matches the structure expected by `data/README_DATA.md`.

## 4) Prepare manifests

```bash
!poetry run python scripts/prepare_manifests.py \
  --data_root /content/drive/MyDrive/italian_parkinson \
  --out_dir data/manifests \
  --split_by speaker
```

## 5) Train a model

```bash
!poetry run python recipes/parkinsons_binary/xvector/train.py \
  recipes/parkinsons_binary/xvector/hparams/train.yaml \
  --data_folder /content/drive/MyDrive/italian_parkinson
```

Swap `xvector` for `ecapa_tdnn`, `wav2vec2`, `wavlm`, or `hubert` to try other recipes.

## 6) Save results to Drive (optional)

Training outputs are written to `results/<model>/<seed>/` inside the Colab VM. To keep them, copy to Drive:

```bash
!cp -r results /content/drive/MyDrive/parkinson-detector-results
```

## Notes

- If `poetry install` is slow, rerun the cell; Colab VMs can be transient.
- For additional options, see `README.md` and the recipe YAML in `recipes/parkinsons_binary/`.
