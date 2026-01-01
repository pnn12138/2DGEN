#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Use project-local cache to avoid permission issues on shared hosts.
export UV_CACHE_DIR="${PROJ_ROOT}/.uv_cache"
mkdir -p "${UV_CACHE_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_CMD=(uv run --no-sync python)

if ! command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=("${PYTHON_BIN}")
fi

usage() {
  cat <<'USAGE'
Run ALIGNN on matbench_jdft2d (exfoliation energy) with Hydra configs.
Options:
  -f, --fold <int>       Fold index for MatbenchCV (default: 0)
  -e, --epochs <int>     Override max epochs (default: config value)
      --fast             Quick smoke run (forces 2 epochs, small batches)
      --cpu              Force CPU training (auto fallback if CUDA unusable)
      --no-tensorboard   Skip launching TensorBoard
      --tb-port <int>    TensorBoard port (default: 6006)
  -h, --help             Show this help message
Examples:
  ./run_alignn_jdft2d.sh              # baseline training with defaults
  ./run_alignn_jdft2d.sh --fold 3     # train fold 3
  ./run_alignn_jdft2d.sh --fast       # quick debug
USAGE
}

FOLD=0
MAX_EPOCHS=""
FAST_RUN=0
USE_CPU=0
LAUNCH_TB=1
TB_PORT=${TB_PORT:-6006}

check_cuda() {
  if [[ ${#PYTHON_CMD[@]} -eq 1 ]] && ! command -v "${PYTHON_CMD[0]}" >/dev/null 2>&1; then
    echo "Python interpreter not found for CUDA check."
    return 1
  fi

  "${PYTHON_CMD[@]}" - <<'PY'
import sys

try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    torch.cuda.get_device_capability(0)
except Exception as exc:  # pragma: no cover - runtime environment check
    print(f"CUDA check failed: {exc}")
    sys.exit(1)

sys.exit(0)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--fold)
      FOLD="$2"
      shift 2
      ;;
    -e|--epochs)
      MAX_EPOCHS="$2"
      shift 2
      ;;
    --fast)
      FAST_RUN=1
      shift
      ;;
    --cpu)
      USE_CPU=1
      shift
      ;;
    --no-tensorboard)
      LAUNCH_TB=0
      shift
      ;;
    --tb-port)
      TB_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "${USE_CPU}" -eq 0 ]]; then
  if ! check_cuda; then
    echo "Falling back to CPU because CUDA is unavailable or unusable."
    USE_CPU=1
  fi
fi

cd "${PROJ_ROOT}"

# Launch TensorBoard if available.
TB_LOGDIR="${PROJ_ROOT}/outputs"
if [[ "${LAUNCH_TB}" -eq 1 ]]; then
  if ! "${PYTHON_CMD[@]}" - <<'PY' >/dev/null 2>&1; then
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("tensorboard") else 1)
PY
    echo "TensorBoard not found; attempting to install via uv (may require network)..."
    if uv pip install --quiet tensorboard >/dev/null 2>&1; then
      echo "TensorBoard installed."
    else
      echo "Warning: failed to install TensorBoard. Proceeding without launching it."
      LAUNCH_TB=0
    fi
  fi

  if [[ "${LAUNCH_TB}" -eq 1 ]]; then
    echo "Starting TensorBoard on port ${TB_PORT}, logdir: ${TB_LOGDIR}"
    uv run --no-sync tensorboard --logdir "${TB_LOGDIR}" --port "${TB_PORT}" --host 0.0.0.0 >/dev/null 2>&1 &
    echo "TensorBoard PID: $!"
  fi
fi

CMD=(
  uv run "${SCRIPT_DIR}/train_alignn_jdft2d.py"
  model=alignn
  data=matbench_jdft2d
  task.fold="${FOLD}"
)

if [[ "${USE_CPU}" -eq 1 ]]; then
  export CUDA_VISIBLE_DEVICES=""
  CMD+=(trainer.trainer.accelerator=cpu)
  CMD+=(trainer.trainer.devices=1)
  CMD+=(trainer.trainer.precision=32)
  CMD+=(data.loader.train.num_workers=0)
  CMD+=(data.loader.val.num_workers=0)
  CMD+=(data.loader.test.num_workers=0)
fi

if [[ -n "${MAX_EPOCHS}" ]]; then
  CMD+=(trainer.trainer.max_epochs="${MAX_EPOCHS}")
fi

if [[ "${FAST_RUN}" -eq 1 ]]; then
  CMD+=(trainer.trainer.max_epochs=2)
  CMD+=(data.loader.train.batch_size=8)
  CMD+=(data.loader.val.batch_size=8)
  CMD+=(data.loader.test.batch_size=8)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
