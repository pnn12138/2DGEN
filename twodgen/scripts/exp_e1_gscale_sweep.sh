#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=path/to/atomdenoiser_last.pt}"
NPZ="${NPZ:-data/C2DB/cache/c2db_tokens_2d_based.npz}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-E1_3_gscale}"
SUMMARY_EXP_ID="${SUMMARY_EXP_ID:-E1_3}"
PROTOCOL="${PROTOCOL:-quick}"
SEEDS="${SEEDS:-0,1,2}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
STEPS="${STEPS:-50}"
COND_MAX="${COND_MAX:-40}"
G_SCALES="${G_SCALES:-0.5,1.0,1.5}"

IFS=',' read -r -a GS_ARR <<< "${G_SCALES}"
for G in "${GS_ARR[@]}"; do
  G_TRIM="$(echo "${G}" | xargs)"
  TOKEN="${G_TRIM//./p}"
  EXP_ID="${EXPERIMENT_PREFIX}_${TOKEN}"
  echo "[run] g_scale=${G_TRIM} -> ${EXP_ID}"
  uv run python -m twodgen.evaluate.ablation_runner \
    --checkpoint "${CHECKPOINT}" \
    --npz "${NPZ}" \
    --runs-root "${RUNS_ROOT}" \
    --experiment-id "${EXP_ID}" \
    --variants full_projection \
    --seeds "${SEEDS}" \
    --protocol "${PROTOCOL}" \
    --num-samples "${NUM_SAMPLES}" \
    --steps "${STEPS}" \
    --cond-max "${COND_MAX}" \
    --sample-args "--g-scale ${G_TRIM} --override-g-scale"
done

uv run python -m twodgen.evaluate.collect_gscale_sweep \
  --runs-root "${RUNS_ROOT}" \
  --experiment-prefix "${EXPERIMENT_PREFIX}" \
  --g-scales "${G_SCALES}" \
  --out "${RUNS_ROOT}/${SUMMARY_EXP_ID}/_aggregate/summary.json"

echo "E1_3 done. Summary: ${RUNS_ROOT}/${SUMMARY_EXP_ID}/_aggregate/summary.json"
