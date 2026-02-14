#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=path/to/atomdenoiser_last.pt}"
NPZ="${NPZ:-data/C2DB/cache/c2db_tokens_2d_based.npz}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-E2_1_}"
SUMMARY_EXP_ID="${SUMMARY_EXP_ID:-E2_1}"
PROTOCOL="${PROTOCOL:-quick}"
SEEDS="${SEEDS:-0,1,2}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
STEPS="${STEPS:-50}"
COND_MAX="${COND_MAX:-40}"
SCHEDULES="${SCHEDULES:-linear,sigmoid,cosine}"
REPULSION_STATES="${REPULSION_STATES:-on,off}"
EXTRA_SAMPLE_ARGS="${EXTRA_SAMPLE_ARGS:-}"

IFS=',' read -r -a SCH_ARR <<< "${SCHEDULES}"
IFS=',' read -r -a REP_ARR <<< "${REPULSION_STATES}"

for SCH_RAW in "${SCH_ARR[@]}"; do
  SCHEDULE="$(echo "${SCH_RAW}" | xargs)"
  if [[ -z "${SCHEDULE}" ]]; then
    continue
  fi
  SCH_UPPER="$(echo "${SCHEDULE}" | tr '[:lower:]' '[:upper:]')"
  CKPT_VAR="CHECKPOINT_${SCH_UPPER}"
  CKPT_USE="${!CKPT_VAR:-${CHECKPOINT}}"
  for REP_RAW in "${REP_ARR[@]}"; do
    REP_STATE="$(echo "${REP_RAW}" | xargs)"
  if [[ "${REP_STATE}" == "on" ]]; then
      REP_ARGS="--min-dist-project"
  elif [[ "${REP_STATE}" == "off" ]]; then
      REP_ARGS="--no-min-dist-project"
  else
      echo "Unsupported repulsion state: ${REP_STATE}" >&2
      exit 1
  fi
    SAMPLE_ARGS="${REP_ARGS}"
    if [[ -n "${EXTRA_SAMPLE_ARGS}" ]]; then
      SAMPLE_ARGS="${SAMPLE_ARGS} ${EXTRA_SAMPLE_ARGS}"
    fi
    EXP_ID="${EXPERIMENT_PREFIX}${SCHEDULE}_rep_${REP_STATE}"
    echo "[run] schedule=${SCHEDULE} repulsion=${REP_STATE} checkpoint=${CKPT_USE}"
    uv run python -m twodgen.evaluate.ablation_runner \
      --checkpoint "${CKPT_USE}" \
      --npz "${NPZ}" \
      --runs-root "${RUNS_ROOT}" \
      --experiment-id "${EXP_ID}" \
      --variants full_projection \
      --seeds "${SEEDS}" \
      --protocol "${PROTOCOL}" \
      --num-samples "${NUM_SAMPLES}" \
      --steps "${STEPS}" \
      --cond-max "${COND_MAX}" \
      --sample-args="${SAMPLE_ARGS}"
  done
done

uv run python -m twodgen.evaluate.collect_e2_curriculum_repulsion \
  --runs-root "${RUNS_ROOT}" \
  --experiment-prefix "${EXPERIMENT_PREFIX}" \
  --out "${RUNS_ROOT}/${SUMMARY_EXP_ID}/_aggregate/summary.json"

echo "E2_1 done. Summary: ${RUNS_ROOT}/${SUMMARY_EXP_ID}/_aggregate/summary.json"
