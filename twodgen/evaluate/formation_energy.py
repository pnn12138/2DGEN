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


def _load_reference(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute formation energies from total energies.")
    parser.add_argument("--energy-jsonl", type=Path, required=True)
    parser.add_argument("--ref-energies", type=Path, required=True, help="JSON mapping element->mu.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--formation-max", type=float, default=0.0)
    parser.add_argument(
        "--missing-strategy",
        type=str,
        choices=["fail", "default"],
        default="fail",
        help="How to handle missing element reference energies.",
    )
    parser.add_argument(
        "--default-mu",
        type=float,
        default=0.0,
        help="Fallback reference energy when missing-strategy=default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.energy_jsonl)
    ref = _load_reference(args.ref_energies)

    per_sample: List[Dict[str, Any]] = []
    pass_flags: List[int] = []
    missing_ref: List[str] = []
    formation_values: List[float] = []

    for row in rows:
        comp = row.get("composition") or {}
        total_energy = row.get("total_energy")
        n_atoms = int(row.get("n_atoms") or 0)
        missing = [el for el in comp.keys() if el not in ref]
        if missing:
            missing_ref.extend(missing)
            if args.missing_strategy == "fail":
                per_sample.append(
                    {
                        "id": row.get("id"),
                        "cif_path": row.get("cif_path"),
                        "formation_energy_per_atom": None,
                        "formation_pass": False,
                        "fail_reason": "missing_ref_energy",
                        "threshold": args.formation_max,
                    }
                )
                pass_flags.append(0)
                continue
        if total_energy is None or n_atoms <= 0:
            per_sample.append(
                {
                    "id": row.get("id"),
                    "cif_path": row.get("cif_path"),
                    "formation_energy_per_atom": None,
                    "formation_pass": False,
                    "fail_reason": "invalid_input",
                    "threshold": args.formation_max,
                }
            )
            pass_flags.append(0)
            continue

        ref_energy = 0.0
        for el, count in comp.items():
            if el in ref:
                mu = float(ref[el])
            else:
                mu = float(args.default_mu)
            ref_energy += mu * int(count)

        formation = (float(total_energy) - ref_energy) / max(n_atoms, 1)
        formation_values.append(formation)
        formation_pass = formation <= float(args.formation_max)
        pass_flags.append(int(formation_pass))
        per_sample.append(
            {
                "id": row.get("id"),
                "cif_path": row.get("cif_path"),
                "formation_energy_per_atom": formation,
                "formation_pass": formation_pass,
                "threshold": args.formation_max,
                "fail_reason": "",
            }
        )

    metrics = {
        "formation_pass_rate": float(np.mean(pass_flags)) if pass_flags else 0.0,
        "formation_energy": {
            "count": len(formation_values),
            "mean": float(np.mean(formation_values)) if formation_values else None,
            "median": float(np.median(formation_values)) if formation_values else None,
        },
        "missing_ref_elements": sorted(set(missing_ref)),
        "missing_strategy": args.missing_strategy,
        "default_mu": args.default_mu if args.missing_strategy == "default" else None,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.out_dir / "per_sample_formation.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    (args.out_dir / "formation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Saved formation energies to {args.out_dir}")


if __name__ == "__main__":
    main()
