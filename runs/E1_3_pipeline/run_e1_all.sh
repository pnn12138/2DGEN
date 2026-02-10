#!/usr/bin/env bash
set -euo pipefail
cd /home/pnn/2dgen

# E1_2 full matrix
CHECKPOINT=outputs/checkpoints/20260207_004939/atomdenoiser_last.pt \
NPZ=data/C2DB/cache/c2db_tokens_2d_based.npz \
RUNS_ROOT=runs \
EXPERIMENT_ID=E1_2 \
PROTOCOL=quick \
SEEDS=0,1,2 \
NUM_SAMPLES=2000 \
STEPS=50 \
COND_MAX=40 \
bash twodgen/scripts/exp_e1_component_ablation.sh

# E1_3 g_scale sweep
CHECKPOINT=outputs/checkpoints/20260207_004939/atomdenoiser_last.pt \
NPZ=data/C2DB/cache/c2db_tokens_2d_based.npz \
RUNS_ROOT=runs \
EXPERIMENT_PREFIX=E1_3_gscale \
SUMMARY_EXP_ID=E1_3 \
PROTOCOL=quick \
SEEDS=0,1,2 \
NUM_SAMPLES=2000 \
STEPS=50 \
COND_MAX=40 \
G_SCALES=0.5,1.0,1.5 \
bash twodgen/scripts/exp_e1_gscale_sweep.sh
