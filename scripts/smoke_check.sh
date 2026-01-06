#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}"

if command -v poetry >/dev/null 2>&1; then
  PYTHON_CMD=(poetry run python)
else
  PYTHON_CMD=(python)
fi

echo "Running smoke checks..."
"${PYTHON_CMD[@]}" scripts/prepare_manifests.py --help >/dev/null
"${PYTHON_CMD[@]}" scripts/predict.py --help >/dev/null
bash -n scripts/run_all.sh
bash -n scripts/download_dataset.sh
for recipe in recipes/parkinsons_binary/*/train.py; do
  "${PYTHON_CMD[@]}" - <<PY >/dev/null
import importlib.util
from pathlib import Path

path = Path("${recipe}")
spec = importlib.util.spec_from_file_location(path.stem, path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
done

echo "Smoke checks passed."
