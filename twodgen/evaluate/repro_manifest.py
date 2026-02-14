from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_registry(path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid registry format: {path}")
    return data.get("experiments", {})


def _run_rows(runs_root: Path, experiment_id: str, config_path: str | None) -> List[Dict[str, Any]]:
    exp_dir = runs_root / experiment_id
    if not exp_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for run_dir in sorted(exp_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "_aggregate":
            continue
        status_path = run_dir / "STATUS.json"
        status_payload = _read_json(status_path) if status_path.exists() else {}
        status = status_payload.get("status", "unknown")
        run_meta_path = run_dir / "run_metadata.json"
        run_meta = _read_json(run_meta_path) if run_meta_path.exists() else {}
        nested_rm = run_meta.get("run_metadata") if isinstance(run_meta, dict) else None
        deps = nested_rm.get("dependencies") if isinstance(nested_rm, dict) else {}
        runtime = nested_rm.get("runtime") if isinstance(nested_rm, dict) else {}
        row = {
            "experiment_id": experiment_id,
            "run_dir": str(run_dir),
            "config": config_path,
            "seed": run_meta.get("seed", status_payload.get("seed")),
            "protocol": run_meta.get("protocol", status_payload.get("protocol")),
            "git_commit": run_meta.get("git_commit", status_payload.get("git_commit")),
            "status": status,
            "inputs": {
                "samples": str(run_dir / "samples" / "samples.npz"),
                "metrics_summary": str(run_dir / "metrics_summary.json"),
                "failure_breakdown": str(run_dir / "failure_breakdown.json"),
            },
            "outputs": {
                "run_metadata": str(run_dir / "run_metadata.json"),
                "projection_stats": str(run_dir / "projection_stats.json"),
            },
            "dependencies": deps,
            "runtime": runtime,
        }
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducibility manifest from experiment registry.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("twodgen/configs/bench/experiments.yaml"),
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = _load_registry(args.registry)
    manifest_rows: List[Dict[str, Any]] = []
    for exp_id, exp_cfg in sorted(experiments.items()):
        config_path = exp_cfg.get("config") if isinstance(exp_cfg, dict) else None
        manifest_rows.extend(_run_rows(args.runs_root, exp_id, config_path))

    payload = {
        "registry": str(args.registry),
        "runs_root": str(args.runs_root),
        "total_runs": len(manifest_rows),
        "runs": manifest_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_id",
                "run_dir",
                "config",
                "seed",
                "protocol",
                "git_commit",
                "status",
                "python",
                "torch",
                "spglib",
                "ase",
                "pymatgen",
                "chgnet",
                "device",
                "dtype",
                "cuda_version",
            ],
        )
        writer.writeheader()
        for row in manifest_rows:
            deps = row.get("dependencies", {}) if isinstance(row, dict) else {}
            runtime = row.get("runtime", {}) if isinstance(row, dict) else {}
            writer.writerow(
                {
                    "experiment_id": row.get("experiment_id"),
                    "run_dir": row.get("run_dir"),
                    "config": row.get("config"),
                    "seed": row.get("seed"),
                    "protocol": row.get("protocol"),
                    "git_commit": row.get("git_commit"),
                    "status": row.get("status"),
                    "python": deps.get("python"),
                    "torch": deps.get("torch"),
                    "spglib": deps.get("spglib"),
                    "ase": deps.get("ase"),
                    "pymatgen": deps.get("pymatgen"),
                    "chgnet": deps.get("chgnet"),
                    "device": runtime.get("device"),
                    "dtype": runtime.get("dtype"),
                    "cuda_version": runtime.get("cuda_version"),
                }
            )
    print(f"Saved reproducibility manifest to {args.out}")
    print(f"Saved reproducibility CSV to {csv_path}")


if __name__ == "__main__":
    main()

