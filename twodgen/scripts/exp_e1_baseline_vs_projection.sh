#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=path/to/atomdenoiser_last.pt}"
NPZ="${NPZ:-data/C2DB/cache/c2db_tokens_2d_based.npz}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
EXPERIMENT_ID="${EXPERIMENT_ID:-E1_1}"
PROTOCOL="${PROTOCOL:-quick}"
SEEDS="${SEEDS:-0,1,2}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
STEPS="${STEPS:-50}"
COND_MAX="${COND_MAX:-40}"
REQUIRE_DELTA="${REQUIRE_DELTA:-}"
EXTRA_SAMPLE_ARGS="${EXTRA_SAMPLE_ARGS:-}"

CMD=(
  uv run python -m twodgen.evaluate.ablation_runner
  --checkpoint "${CHECKPOINT}"
  --npz "${NPZ}"
  --runs-root "${RUNS_ROOT}"
  --experiment-id "${EXPERIMENT_ID}"
  --variants baseline,full_projection
  --seeds "${SEEDS}"
  --protocol "${PROTOCOL}"
  --num-samples "${NUM_SAMPLES}"
  --steps "${STEPS}"
  --cond-max "${COND_MAX}"
)

if [[ -n "${EXTRA_SAMPLE_ARGS}" ]]; then
  CMD+=(--sample-args "${EXTRA_SAMPLE_ARGS}")
fi
if [[ -n "${REQUIRE_DELTA}" ]]; then
  CMD+=(--require-delta "${REQUIRE_DELTA}")
fi

"${CMD[@]}"

echo "E1_1 done. Summary: ${RUNS_ROOT}/${EXPERIMENT_ID}/_aggregate/summary.json"
