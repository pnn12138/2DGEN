from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _parse_steps(value: str) -> List[str]:
    steps = [v.strip() for v in value.split(",") if v.strip()]
    return steps if steps else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIF -> Tier-0/conditions/MatterSim/formation pipeline.")
    parser.add_argument("--cif-dir", type=Path, default=None)
    parser.add_argument("--cif-list", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-dist-cut", type=float, default=1.5)
    parser.add_argument("--bond-cut", type=float, default=3.0)
    parser.add_argument("--dup-eps", type=float, default=1e-3)
    parser.add_argument("--vacuum-min", type=float, default=None)
    parser.add_argument("--vacuum-ratio-min", type=float, default=None)
    parser.add_argument("--pbc-mask", type=str, default="1,1,0")
    parser.add_argument("--target-formula", type=str, default=None)
    parser.add_argument("--target-elements", type=str, default=None)
    parser.add_argument("--target-spacegroup", type=int, default=None)
    parser.add_argument("--mattersim-model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ref-energies", type=Path, required=False)
    parser.add_argument("--formation-max", type=float, default=0.0)
    parser.add_argument(
        "--pipeline-steps",
        type=str,
        default="tier0,conditions,energy,merge",
        help="Comma-separated steps to execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = _parse_steps(args.pipeline_steps)

    tier0_dir = out_dir / "tier0"
    cond_dir = out_dir / "conditions"
    energy_dir = out_dir / "energy"
    merged_dir = out_dir / "merged"

    base_inputs = []
    if args.cif_dir is not None:
        base_inputs += ["--cif-dir", str(args.cif_dir)]
    if args.cif_list is not None:
        base_inputs += ["--cif-list", str(args.cif_list)]

    if "tier0" in steps:
        _run(
            [
                "python",
                "-m",
                "twodgen.evaluate.eval_tier0_cif",
                *base_inputs,
                "--out-dir",
                str(tier0_dir),
                "--min-dist-cut",
                str(args.min_dist_cut),
                "--bond-cut",
                str(args.bond_cut),
                "--dup-eps",
                str(args.dup_eps),
                "--pbc-mask",
                args.pbc_mask,
            ]
            + (["--vacuum-min", str(args.vacuum_min)] if args.vacuum_min is not None else [])
            + (
                ["--vacuum-ratio-min", str(args.vacuum_ratio_min)]
                if args.vacuum_ratio_min is not None
                else []
            )
        )

    if "conditions" in steps:
        _run(
            [
                "python",
                "-m",
                "twodgen.evaluate.check_conditions",
                *base_inputs,
                "--out-dir",
                str(cond_dir),
            ]
            + (["--target-formula", args.target_formula] if args.target_formula else [])
            + (["--target-elements", args.target_elements] if args.target_elements else [])
            + (["--target-spacegroup", str(args.target_spacegroup)] if args.target_spacegroup else [])
        )

    need_energy = "energy" in steps or "formation" in steps
    need_formation = "formation" in steps
    if need_formation and args.ref_energies is None:
        raise ValueError("--ref-energies is required when pipeline includes formation.")

    if need_energy:
        _run(
            [
                "python",
                "-m",
                "twodgen.evaluate.mattersim_energy",
                *base_inputs,
                "--out-dir",
                str(energy_dir),
                "--device",
                args.device,
                "--fmax",
                str(args.fmax),
                "--steps",
                str(args.steps),
            ]
            + (["--model-path", args.mattersim_model] if args.mattersim_model else [])
            + (["--relax"] if args.relax else [])
            + (["--ref-energies", str(args.ref_energies)] if args.ref_energies else [])
            + (["--formation-max", str(args.formation_max)] if need_formation else [])
        )

    if "merge" in steps:
        merge_cmd = [
            "python",
            "-m",
            "twodgen.evaluate.merge_reports",
            "--tier0",
            str(tier0_dir / "per_sample_tier0.jsonl"),
            "--conditions",
            str(cond_dir / "per_sample_conditions.jsonl"),
            "--energy",
            str(energy_dir / "per_sample_energy.jsonl"),
        ]
        if need_formation:
            merge_cmd += [
                "--formation",
                str(energy_dir / "per_sample_formation.jsonl"),
            ]
        _run(
            merge_cmd
            + [
                "--out-dir",
                str(merged_dir),
            ]
        )

    pipeline_summary = {
        "tier0": str(tier0_dir),
        "conditions": str(cond_dir),
        "energy": str(energy_dir),
        "formation": str(energy_dir) if need_formation else None,
        "merged": str(merged_dir),
        "steps": steps,
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(pipeline_summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"Pipeline outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
