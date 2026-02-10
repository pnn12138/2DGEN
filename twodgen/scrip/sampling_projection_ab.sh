#!/usr/bin/env bash
# A/B sampling projection regression:
# A: projection off
# B: post-step projection on (angle/cond/inplane/volume) every step (interval=1)
#
# Usage:
#   OUT_ROOT=outputs/ab_proj CHECKPOINT=... NPZ=... bash twodgen/scrip/sampling_projection_ab.sh

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-outputs/ab_sampling_projection}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=path/to/atomdenoiser_best.pt}"
NPZ="${NPZ:-data/C2DB/cache/c2db_tokens_2d_based.npz}"
NUM="${NUM:-128}"
STEPS="${STEPS:-50}"
SEED="${SEED:-123}"

COND_MAX="${COND_MAX:-40}"

BASE_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --npz "${NPZ}"
  --num-samples "${NUM}"
  --steps "${STEPS}"
  --seed "${SEED}"
  --project-gram-cond
  --project-gram-max-cond "${COND_MAX}"
  --project-final
)

echo "[A] projection off -> ${OUT_ROOT}/A"
UV_CACHE_DIR="${UV_CACHE_DIR:-uv_cache}" PYTHONPATH="$(pwd)" uv run python -m twodgen.scrip.sample_tokens \
  "${BASE_ARGS[@]}" \
  --no-post-project \
  --out-dir "${OUT_ROOT}/A"

echo "[B] post-step projection on -> ${OUT_ROOT}/B"
UV_CACHE_DIR="${UV_CACHE_DIR:-uv_cache}" PYTHONPATH="$(pwd)" uv run python -m twodgen.scrip.sample_tokens \
  "${BASE_ARGS[@]}" \
  --post-project --post-project-interval 1 --post-project-keys "angle,cond,inplane,volume" \
  --post-project-cond-max "${COND_MAX}" \
  --out-dir "${OUT_ROOT}/B"

echo "Done."
