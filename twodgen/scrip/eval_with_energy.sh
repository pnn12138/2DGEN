#!/usr/bin/env bash
# Sample + relax + eval with energy taxonomy.
#
# Usage:
#   OUT_DIR=outputs/eval_energy CHECKPOINT=... NPZ=... bash twodgen/scrip/eval_with_energy.sh

set -euo pipefail

OUT_DIR="${OUT_DIR:-outputs/eval_with_energy}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=path/to/atomdenoiser_best.pt}"
NPZ="${NPZ:-data/C2DB/cache/c2db_tokens_2d_based.npz}"
NUM="${NUM:-64}"
STEPS="${STEPS:-50}"
SEED="${SEED:-123}"
COND_MAX="${COND_MAX:-40}"

UV_CACHE_DIR="${UV_CACHE_DIR:-uv_cache}" PYTHONPATH="$(pwd)" uv run python - <<'PY'
import importlib
missing = []
for m in ("chgnet","ase"):
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(f"Missing deps for --relax: {missing}. Install them in the uv env first.")
print("[info] deps ok: chgnet/ase")
PY

echo "[sample+relax] -> ${OUT_DIR}"
UV_CACHE_DIR="${UV_CACHE_DIR:-uv_cache}" PYTHONPATH="$(pwd)" uv run python -m twodgen.scrip.sample_tokens \
  --checkpoint "${CHECKPOINT}" --npz "${NPZ}" \
  --num-samples "${NUM}" --steps "${STEPS}" --seed "${SEED}" \
  --project-gram-cond --project-gram-max-cond "${COND_MAX}" --project-final \
  --post-project --post-project-interval 1 --post-project-keys "angle,cond,inplane,volume" \
  --relax --relax-steps 30 --relax-fmax 0.1 --relax-device cuda \
  --out-dir "${OUT_DIR}"

echo "[eval] -> ${OUT_DIR}/eval"
UV_CACHE_DIR="${UV_CACHE_DIR:-uv_cache}" PYTHONPATH="$(pwd)" uv run python -m twodgen.evaluate.eval_samples \
  --samples "${OUT_DIR}/samples.npz" --out-dir "${OUT_DIR}/eval" --cond-max "${COND_MAX}"

echo "Done."
