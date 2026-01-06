#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-data/raw/italian_parkinson}
ZIP_PATH=${2:-data/raw/italian_parkinson.zip}

URL="https://huggingface.co/datasets/birgermoell/Italian_Parkinsons_Voice_and_Speech/resolve/main/italian_parkinson/Italian%20Parkinson's%20Voice%20and%20speech.zip?download=true"

mkdir -p "$(dirname "${ZIP_PATH}")"

if [[ -d "${DATA_ROOT}/15 Young Healthy Control" && -d "${DATA_ROOT}/22 Elderly Healthy Control" && -d "${DATA_ROOT}/28 People with Parkinson's disease" ]]; then
  echo "Dataset already present at ${DATA_ROOT}"
  exit 0
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Downloading dataset to ${ZIP_PATH}..."
  curl -L "${URL}" -o "${ZIP_PATH}"
else
  echo "Using existing archive at ${ZIP_PATH}"
fi

echo "Extracting to ${DATA_ROOT}..."
mkdir -p "${DATA_ROOT}"
unzip -q "${ZIP_PATH}" -d "${DATA_ROOT}"

echo "Done."
