from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-g_scale ablation summaries into one table.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment-prefix", type=str, default="E1_3_gscale")
    parser.add_argument("--g-scales", type=str, required=True, help="Comma-separated g_scale values.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    g_scales = [s.strip() for s in args.g_scales.split(",") if s.strip()]
    rows: List[Dict[str, Any]] = []
    for g in g_scales:
        token = g.replace(".", "p")
        exp_id = f"{args.experiment_prefix}_{token}"
        summary_path = args.runs_root / exp_id / "_aggregate" / "summary.json"
        if not summary_path.exists():
            rows.append({"g_scale": float(g), "available": False})
            continue
        summary = _read_json(summary_path)
        full = summary.get("variants", {}).get("full_projection", {})
        metric = full.get("success_geom_rate", {})
        rows.append(
            {
                "g_scale": float(g),
                "available": True,
                "experiment_id": exp_id,
                "success_geom_rate_mean": metric.get("mean"),
                "success_geom_rate_std": metric.get("std"),
                "valid_rate_eval_mean": full.get("valid_rate_eval", {}).get("mean"),
                "post_project_trigger_any_rate_mean": full.get("post_project_trigger_any_rate", {}).get("mean"),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2, ensure_ascii=True)
    csv = ["g_scale,available,experiment_id,success_geom_rate_mean,success_geom_rate_std,valid_rate_eval_mean,post_project_trigger_any_rate_mean"]
    for row in rows:
        csv.append(
            f"{row.get('g_scale')},{row.get('available')},{row.get('experiment_id','')},"
            f"{row.get('success_geom_rate_mean','')},{row.get('success_geom_rate_std','')},"
            f"{row.get('valid_rate_eval_mean','')},{row.get('post_project_trigger_any_rate_mean','')}"
        )
    args.out.with_suffix(".csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    print(f"Wrote g_scale sweep summary: {args.out}")


if __name__ == "__main__":
    main()

