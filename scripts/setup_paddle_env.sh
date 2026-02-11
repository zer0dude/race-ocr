#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv-paddle}"

# Paddle GPU wheel index for CUDA 11.8 builds
PADDLE_INDEX="${PADDLE_INDEX:-https://www.paddlepaddle.org.cn/packages/stable/cu118/}"

REQ_IN="${REQ_IN:-requirements-paddle.in}"

echo "[1/10] Creating venv at: ${VENV_DIR}"
${PYTHON_BIN} -m venv "${VENV_DIR}"

echo "[2/10] Activating venv"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[3/10] Upgrading pip tooling"
python -m pip install -U pip wheel setuptools

echo "[4/10] Installing PaddlePaddle GPU (CUDA 11.8 build) from Paddle index"
python -m pip install "paddlepaddle-gpu==3.2.0" -i "${PADDLE_INDEX}"

echo "[5/10] Installing OCR dependencies (${REQ_IN})"
python -m pip install -r "${REQ_IN}"

echo "[6/10] Force Torch to CPU-only (prevents CUDA/NCCL conflicts if a dependency pulls torch)"
python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "[7/10] Patch venv activate to export CUDA runtime libs bundled as wheels (LD_LIBRARY_PATH)"
ACTIVATE_FILE="${VENV_DIR}/bin/activate"
MARKER_BEGIN="# >>> race-ocr paddle ld_library_path >>>"
MARKER_END="# <<< race-ocr paddle ld_library_path <<<"

if ! grep -qF "${MARKER_BEGIN}" "${ACTIVATE_FILE}"; then
  SITEPKG="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
  cat >> "${ACTIVATE_FILE}" <<EOF

${MARKER_BEGIN}
# Ensure Paddle can find CUDA/cuDNN/NCCL libs shipped as pip wheels.
# (Needed on minimal OS images where /usr/lib doesn't include cuDNN.)
_SITEPKG="${SITEPKG}"
export LD_LIBRARY_PATH="\${_SITEPKG}/nvidia/cudnn/lib:\${_SITEPKG}/nvidia/cublas/lib:\${_SITEPKG}/nvidia/cuda_runtime/lib:\${_SITEPKG}/nvidia/nccl/lib:\${LD_LIBRARY_PATH:-}"
unset _SITEPKG
${MARKER_END}
EOF
  echo "  - Patched ${ACTIVATE_FILE}"
else
  echo "  - activate already patched"
fi

# Re-source activate so current shell inherits the new exports too
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[8/10] Smoke tests: paddle + gpu"
python - <<'PY'
import paddle
print("paddle:", paddle.__version__)
print("compiled_with_cuda:", paddle.is_compiled_with_cuda())
print("device:", paddle.device.get_device())
print("paddle cuda version:", paddle.version.cuda())
print("paddle cudnn version:", paddle.version.cudnn())
PY

echo "[9/10] Smoke tests: PaddleOCR init (GPU)"
# Skip slow “model hoster connectivity” checks during smoke tests
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

python - <<'PY'
from paddleocr import PaddleOCR

# PaddleOCR v3.x uses a pipeline-style API; GPU is selected via `device`, not `use_gpu`.
# `use_angle_cls` is deprecated; use `use_textline_orientation`.
ocr = PaddleOCR(
    lang="en",
    device="gpu:0",
    use_textline_orientation=True,
)
print("PaddleOCR init OK (device=gpu:0)")
PY

echo "[10/10] Done."
echo "✅ Paddle OCR env setup complete."
