from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_registry(path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid registry: {path}")
    experiments = data.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError(f"Registry missing experiments: {path}")
    return experiments


def _metric_from_summary(summary: Dict[str, Any], key: str) -> float | None:
    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        entry = metrics.get(key)
        if isinstance(entry, dict) and isinstance(entry.get("mean"), (int, float)):
            return float(entry["mean"])

    variants = summary.get("variants")
    if isinstance(variants, dict):
        for variant_name in ("full_projection", "baseline"):
            var = variants.get(variant_name)
            if isinstance(var, dict):
                entry = var.get(key)
                if isinstance(entry, dict) and isinstance(entry.get("mean"), (int, float)):
                    return float(entry["mean"])

    rows = summary.get("rows")
    if isinstance(rows, list) and rows:
        vals = [r.get(key) for r in rows if isinstance(r, dict) and isinstance(r.get(key), (int, float))]
        if vals:
            return float(sum(float(v) for v in vals) / len(vals))
    return None


def _build_exp_row(exp_id: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "success_geom_rate": _metric_from_summary(summary, "success_geom_rate"),
        "valid_rate_eval": _metric_from_summary(summary, "valid_rate_eval"),
        "post_project_trigger_any_rate": _metric_from_summary(summary, "post_project_trigger_any_rate"),
        "bad_volume_rate": _metric_from_summary(summary, "bad_volume_rate"),
        "spacegroup_match_rate": _metric_from_summary(summary, "spacegroup_match_rate"),
        "spglib_fail_rate": _metric_from_summary(summary, "spglib_fail_rate"),
        "energy_available_rate": _metric_from_summary(summary, "energy_available_rate"),
        "novelty_mean": _metric_from_summary(summary, "novelty_mean"),
    }


def _write_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id",
        "success_geom_rate",
        "valid_rate_eval",
        "post_project_trigger_any_rate",
        "bad_volume_rate",
        "spacegroup_match_rate",
        "spglib_fail_rate",
        "energy_available_rate",
        "novelty_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_figure(fig_id: str, path_png: Path, rows: List[Dict[str, Any]]) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    x = []
    y = []
    labels = []
    for row in rows:
        xv = row.get("success_geom_rate")
        yv = row.get("valid_rate_eval")
        if isinstance(xv, (int, float)) and isinstance(yv, (int, float)):
            x.append(float(xv))
            y.append(float(yv))
            labels.append(str(row.get("experiment_id", "")))
    fig, ax = plt.subplots(figsize=(6, 4))
    if x:
        ax.scatter(x, y, s=60)
        for xv, yv, label in zip(x, y, labels):
            ax.text(xv, yv, label, fontsize=8)
    ax.set_xlabel("success_geom_rate")
    ax.set_ylabel("valid_rate_eval")
    ax.set_title(f"{fig_id}: Validity vs Geometry Success")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_png, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper tables/figures from experiment registry.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("twodgen/configs/bench/experiments.yaml"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = _load_registry(args.registry)
    summaries: Dict[str, Dict[str, Any]] = {}
    paper_map: Dict[str, List[str]] = {}
    for exp_id, cfg in sorted(registry.items()):
        summary_path = args.runs_root / exp_id / "_aggregate" / "summary.json"
        if summary_path.exists():
            summaries[exp_id] = _read_json(summary_path)
        assets = cfg.get("paper_assets", []) if isinstance(cfg, dict) else []
        if isinstance(assets, list):
            for asset in assets:
                paper_map.setdefault(str(asset), []).append(exp_id)

    rows_by_exp: Dict[str, Dict[str, Any]] = {
        exp_id: _build_exp_row(exp_id, summary)
        for exp_id, summary in summaries.items()
    }
    out_dir = args.out_dir
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for table_id in ("Table1", "Table2", "Table3"):
        exp_ids = paper_map.get(table_id, [])
        rows = [rows_by_exp[e] for e in exp_ids if e in rows_by_exp]
        _write_table(tables_dir / f"{table_id}.csv", rows)

    for fig_id in ("Fig2", "Fig3", "Fig4", "Fig5", "Fig6"):
        exp_ids = paper_map.get(fig_id, [])
        rows = [rows_by_exp[e] for e in exp_ids if e in rows_by_exp]
        (data_dir / f"{fig_id}.json").write_text(
            json.dumps({"figure": fig_id, "rows": rows}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        _write_figure(fig_id, figures_dir / f"{fig_id}.png", rows)

    manifest = {
        "registry": str(args.registry),
        "runs_root": str(args.runs_root),
        "tables": [str(p) for p in sorted(tables_dir.glob("*.csv"))],
        "figures": [str(p) for p in sorted(figures_dir.glob("*.png"))],
        "figure_data": [str(p) for p in sorted(data_dir.glob("*.json"))],
    }
    (out_dir / "paper_assets_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"Exported paper assets to {out_dir}")


if __name__ == "__main__":
    main()

