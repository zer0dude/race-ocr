#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-yolo}"

echo "[1/6] Creating venv at: ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"

echo "[2/6] Activating venv"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[3/6] Upgrading pip tooling"
python -m pip install -U pip wheel setuptools

echo "[4/6] Installing PyTorch (CUDA 11.8) from official PyTorch index"
# Tesla T4 is happy with cu118 builds
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[5/6] Installing YOLO dependencies"
pip install -r requirements-yolo.in

echo "[6/6] Smoke tests"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
import ultralytics
print("ultralytics:", ultralytics.__version__)
PY

echo "✅ YOLO env setup complete."
