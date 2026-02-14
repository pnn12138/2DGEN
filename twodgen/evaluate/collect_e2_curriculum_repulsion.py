from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_exp_label(exp_id: str, prefix: str) -> Tuple[Optional[str], Optional[str]]:
    if not exp_id.startswith(prefix):
        return None, None
    suffix = exp_id[len(prefix) :]
    if "_rep_" not in suffix:
        return None, None
    schedule, repulsion = suffix.split("_rep_", 1)
    schedule = schedule.strip() or None
    repulsion = repulsion.strip() or None
    return schedule, repulsion


def _safe_mean(summary: Dict[str, Any], metric: str) -> Optional[float]:
    variants = summary.get("variants", {})
    full = variants.get("full_projection", {})
    value = full.get(metric, {})
    if isinstance(value, dict):
        mean_v = value.get("mean")
        if isinstance(mean_v, (int, float)):
            return float(mean_v)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect E2 curriculum/repulsion ablation summaries.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment-prefix", type=str, default="E2_1_")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root
    rows: List[Dict[str, Any]] = []
    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_id = exp_dir.name
        schedule, repulsion = _parse_exp_label(exp_id, args.experiment_prefix)
        if schedule is None or repulsion is None:
            continue
        summary_path = exp_dir / "_aggregate" / "summary.json"
        if not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        rows.append(
            {
                "experiment_id": exp_id,
                "schedule": schedule,
                "repulsion": repulsion,
                "available": True,
                "success_geom_rate": _safe_mean(summary, "success_geom_rate"),
                "post_project_trigger_any_rate": _safe_mean(summary, "post_project_trigger_any_rate"),
                "valid_rate_eval": _safe_mean(summary, "valid_rate_eval"),
                "collision_rate": _safe_mean(summary, "collision_rate"),
                "summary_path": str(summary_path),
            }
        )

    rows = sorted(rows, key=lambda r: (str(r.get("schedule")), str(r.get("repulsion"))))
    payload = {
        "experiment_prefix": args.experiment_prefix,
        "rows": rows,
        "total_rows": len(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved E2 summary to {args.out}")


if __name__ == "__main__":
    main()

