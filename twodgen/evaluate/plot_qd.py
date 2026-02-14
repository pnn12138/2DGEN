from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _diversity_score(div: Dict[str, Any]) -> float:
    vals = []
    for key in ("spacegroup", "n_atoms_coverage", "lattice_coverage"):
        obj = div.get(key, {})
        if key == "spacegroup":
            v = obj.get("coverage_vs_230")
        else:
            v = obj.get("coverage")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validity-diversity (QD) scatter.")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="Entries in label:metrics_summary.json:diversity.json format.",
    )
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for item in args.runs:
        parts = item.split(":", 2)
        if len(parts) != 3:
            raise ValueError("Each --runs item must be label:metrics_summary:diversity")
        label, metrics_path, diversity_path = parts
        metrics = _read_json(Path(metrics_path))
        diversity = _read_json(Path(diversity_path))
        valid = metrics.get("success_geom_rate")
        if valid is None:
            valid = metrics.get("valid_rate_eval")
        valid = float(valid) if isinstance(valid, (int, float)) else 0.0
        div_score = _diversity_score(diversity)
        rows.append(
            {
                "label": label,
                "validity": valid,
                "diversity": div_score,
                "metrics_path": metrics_path,
                "diversity_path": diversity_path,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["label,validity,diversity,metrics_path,diversity_path"]
    for row in rows:
        lines.append(
            f"{row['label']},{row['validity']},{row['diversity']},{row['metrics_path']},{row['diversity_path']}"
        )
    args.out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in rows:
        ax.scatter(row["validity"], row["diversity"], s=60)
        ax.text(row["validity"], row["diversity"], row["label"], fontsize=8)
    ax.set_xlabel("Validity (success_geom_rate)")
    ax.set_ylabel("Diversity coverage")
    ax.set_title("Validity-Diversity Tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)
    plt.close(fig)
    print(f"Saved QD plot to {args.out_png}")


if __name__ == "__main__":
    main()

