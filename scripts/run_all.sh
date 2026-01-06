#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-data/raw/italian_parkinson}
MANIFEST_DIR=data/manifests

export PYTHONPATH="src:${PYTHONPATH:-}"

poetry run python scripts/prepare_manifests.py --data_root "${DATA_ROOT}" --out_dir "${MANIFEST_DIR}" --split_by speaker

models=(xvector ecapa_tdnn wav2vec2 wavlm hubert)

for model in "${models[@]}"; do
  hp="recipes/parkinsons_binary/${model}/hparams/train.yaml"
  train_py="recipes/parkinsons_binary/${model}/train.py"
  echo "=== Training ${model} ==="
  poetry run python "${train_py}" "${hp}" --data_folder "${DATA_ROOT}" --device "${DEVICE:-cpu}"
done

echo "Runs finished. Update reports/results.md with metrics from logs in results/<model>/<seed>/train_log.txt."
