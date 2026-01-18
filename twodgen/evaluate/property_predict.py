from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Placeholder property predictor.")
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--property-key", type=str, default="band_gap")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["heuristic", "constant", "random"],
        default="heuristic",
    )
    parser.add_argument("--value", type=float, default=1.0)
    parser.add_argument("--min", dest="min_val", type=float, default=0.0)
    parser.add_argument("--max", dest="max_val", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vacuum-weight",
        type=float,
        default=0.05,
        help="Weight applied to the vacuum thickness when predicting the property.",
    )
    parser.add_argument(
        "--thickness-weight",
        type=float,
        default=0.02,
        help="Weight applied to slab thickness when predicting the property.",
    )
    parser.add_argument(
        "--min-dist-weight",
        type=float,
        default=0.1,
        help="Weight applied to min_dist when predicting the property.",
    )
    parser.add_argument(
        "--cross-vacuum-penalty",
        type=float,
        default=-0.25,
        help="Penalty added when a cross-vacuum bond is detected.",
    )
    parser.add_argument(
        "--valid-bonus",
        type=float,
        default=0.4,
        help="Bonus added when the sample already passes Tier-0 validity checks.",
    )
    parser.add_argument(
        "--mock-predict",
        action="store_true",
        help="Enable mock predictions (forces random mode).",
    )
    return parser.parse_args()



def _safe_float(value: Any) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    return val


def _heuristic_predict(row: Dict[str, Any], args: argparse.Namespace) -> float:
    base = float(args.value)
    vacuum = _safe_float(row.get("vacuum"))
    if vacuum is not None:
        base += vacuum * args.vacuum_weight
    thickness = _safe_float(row.get("thickness"))
    if thickness is not None:
        base += thickness * args.thickness_weight
    min_dist = _safe_float(row.get("min_dist"))
    if min_dist is not None:
        base += min_dist * args.min_dist_weight
    if row.get("cross_vacuum_bond"):
        base += args.cross_vacuum_penalty
    valid = bool(row.get("valid"))
    if valid:
        base += args.valid_bonus
    if bool(row.get("valid_2d")):
        base += args.valid_bonus * 0.6
    return float(np.clip(base, args.min_val, args.max_val))


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.per_sample)
    rng = np.random.default_rng(args.seed)
    mode = "random" if args.mock_predict else args.mode

    per_sample: List[Dict[str, Any]] = []
    pass_flags: List[int] = []
    values: List[float] = []

    for row in rows:
        if mode == "random":
            value = float(rng.uniform(args.min_val, args.max_val))
        elif mode == "constant":
            value = float(np.clip(args.value, args.min_val, args.max_val))
        else:
            value = _heuristic_predict(row, args)
        passed = value >= float(args.threshold)
        pass_flags.append(int(passed))
        values.append(value)
        per_sample.append(
            {
                "id": row.get("id"),
                "cif_path": row.get("cif_path"),
                args.property_key: value,
                "property_pass": passed,
            }
        )

    metrics = {
        "property_key": args.property_key,
        "pass_rate": float(np.mean(pass_flags)) if pass_flags else 0.0,
        "value_mean": float(np.mean(values)) if values else None,
        "mode": mode,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.out_dir / "per_sample_property.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    (args.out_dir / "property_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Saved property predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
