#!/usr/bin/env bash
# Reproducible cond-trigger short run (E0). Set output root once and keep seed/batch/log_interval identical.
# Usage: bash twodgen/scrip/debug_cond_trigger.sh

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-outputs/debug_cond_trigger}"
NPZ="${NPZ:-data/C2DB/cache/c2db_tokens_2d_based.npz}"
SPLIT_JSON="${SPLIT_JSON:-data/C2DB/cache/c2db_tokens_split.json}"
SEED=123
BATCH=64
LOG_INT=10
COND_MAX=40
WARMUP=80
NUM_WORKERS=2

echo "[E0-on] cond loss enabled -> ${OUT_ROOT}/on"
UV_CACHE_DIR="${UV_CACHE_DIR:-uv_cache}" PYTHONPATH="$(pwd)" uv run python twodgen/scrip/train_tokens.py \
  --npz "${NPZ}" --split-json "${SPLIT_JSON}" --split train \
  --epochs 1 --batch-size "${BATCH}" --log-interval "${LOG_INT}" \
  --num-workers "${NUM_WORKERS}" --seed "${SEED}" \
  --cond-max "${COND_MAX}" --cond-loss-weight 0.1 \
  --loss-weight-warmup-steps "${WARMUP}" --loss-weight-warmup-start 0 --loss-weight-warmup-end 1 \
  --save-dir "${OUT_ROOT}/on"

echo "[E0-off] cond loss disabled -> ${OUT_ROOT}/off"
UV_CACHE_DIR="${UV_CACHE_DIR:-uv_cache}" PYTHONPATH="$(pwd)" uv run python twodgen/scrip/train_tokens.py \
  --npz "${NPZ}" --split-json "${SPLIT_JSON}" --split train \
  --epochs 1 --batch-size "${BATCH}" --log-interval "${LOG_INT}" \
  --num-workers "${NUM_WORKERS}" --seed "${SEED}" \
  --cond-max "${COND_MAX}" --cond-loss-weight 0 \
  --loss-weight-warmup-steps "${WARMUP}" --loss-weight-warmup-start 0 --loss-weight-warmup-end 1 \
  --save-dir "${OUT_ROOT}/off"

cat <<'NOTE'
验收字段（每个 log_interval 输出）：
- loss_cond_number（on>0，off=0）
- cond_gram_mean/p50/p95/max、cond_lattice_mean/p50/p95/max
- cond_gram_violation_rate、cond_lattice_violation_rate
- cond_gram_lattice_spearman（相关性）

对照口径：
1) on 组 p95/max 需显著低于 off 组；violation rate 下降
2) gram vs lattice 至少单调相关；若 gram 下降而 lattice 仍爆炸，需检查口径/作用点
NOTE
