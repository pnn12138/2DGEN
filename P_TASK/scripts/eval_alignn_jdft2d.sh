#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT_PATH=${1:-}
FOLD=${2:-0}

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Usage: $0 <checkpoint_path> [fold]"
  exit 1
fi

uv run "${SCRIPT_DIR}/eval_alignn_jdft2d.py" \
  model=alignn \
  data=matbench_jdft2d \
  task.fold="${FOLD}" \
  task.checkpoint_path="${CKPT_PATH}" \
  data.loader.test.batch_size=8
