from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

from twodgen.evaluate.run_layout import (
    METRICS_SUMMARY_SCHEMA_VERSION,
    atomic_write_json,
    make_schema_payload,
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_metric_files(exp_dir: Path) -> List[Path]:
    files: List[Path] = []
    if not exp_dir.exists():
        return files
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "_aggregate":
            continue
        path = run_dir / "metrics_summary.json"
        if path.exists():
            files.append(path)
    return files


def _ci95(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return 1.96 * stdev(values) / math.sqrt(len(values))


def _aggregate_numeric(metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys: set[str] = set()
    for row in metrics:
        for key, value in row.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                keys.add(key)
    out: Dict[str, Dict[str, float]] = {}
    for key in sorted(keys):
        values = [float(row[key]) for row in metrics if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))]
        if not values:
            continue
        out[key] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "count": float(len(values)),
        }
        ci = _ci95(values)
        if ci is not None:
            out[key]["ci95"] = ci
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate runs/<EXP>/*/metrics_summary.json to _aggregate.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--protocol", type=str, default=None)
    args = parser.parse_args()

    exp_dir = args.runs_root / args.experiment_id
    metric_files = _find_metric_files(exp_dir)
    if not metric_files:
        raise FileNotFoundError(f"No metrics_summary.json found under {exp_dir}")

    per_run: List[Dict[str, Any]] = []
    seeds: List[int] = []
    for path in metric_files:
        row = _read_json(path)
        row["_run_dir"] = str(path.parent)
        per_run.append(row)
        seed = row.get("seed")
        if isinstance(seed, int):
            seeds.append(seed)

    aggregate = {
        "experiment_id": args.experiment_id,
        "num_runs": len(per_run),
        "seed_count": len(set(seeds)),
        "metrics": _aggregate_numeric(per_run),
        "runs": [{"path": row["_run_dir"], "seed": row.get("seed")} for row in per_run],
    }
    aggregate = make_schema_payload(
        schema_version=METRICS_SUMMARY_SCHEMA_VERSION,
        payload=aggregate,
        experiment_id=args.experiment_id,
        seed=None,
        protocol=args.protocol,
        config_hash_value=None,
    )

    out_dir = exp_dir / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "summary.json", aggregate)

    csv_lines = ["metric,mean,std,ci95,count"]
    for key, stats in aggregate["metrics"].items():
        csv_lines.append(
            f"{key},{stats.get('mean','')},{stats.get('std','')},{stats.get('ci95','')},{stats.get('count','')}"
        )
    (out_dir / "summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Wrote aggregate summary to {out_dir}")


if __name__ == "__main__":
    main()

