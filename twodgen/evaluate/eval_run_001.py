from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from twodgen.common.run_metadata import collect_run_metadata
from twodgen.scrip import sample_tokens as sample_tokens_mod


REPORT_SCHEMA_VERSION = "eval_run_001_v1"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible baseline eval_run_001 runner.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to atomdenoiser checkpoint.")
    parser.add_argument("--npz", type=Path, required=True, help="Token cache npz (conditioning + stats).")
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Split json produced by twodgen.data.create_c2db_split.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/eval_run_001"))
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-cif", action="store_true", help="Also export CIF files (can be large).")
    parser.add_argument("--eval-min-dist", type=float, default=1.5)
    parser.add_argument("--eval-bond-cut", type=float, default=3.0)
    parser.add_argument("--eval-dup-eps", type=float, default=1e-3)
    return parser.parse_args(argv)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_one(
    *,
    scope_name: str,
    cond_split: str,
    checkpoint: Path,
    npz: Path,
    split_json: Path,
    out_dir: Path,
    num_samples: int,
    steps: int,
    method: str,
    seed: int,
    save_cif: bool,
    eval_min_dist: float,
    eval_bond_cut: float,
    eval_dup_eps: float,
) -> Dict[str, Any]:
    scope_dir = out_dir / scope_name
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--npz",
        str(npz),
        "--cond-npz",
        str(npz),
        "--cond-split-json",
        str(split_json),
        "--cond-split",
        str(cond_split),
        "--num-samples",
        str(num_samples),
        "--steps",
        str(steps),
        "--method",
        str(method),
        "--out-dir",
        str(scope_dir),
        "--seed",
        str(seed),
    ]
    sample_args = sample_tokens_mod.parse_args(argv)
    sample_args.save_cif = bool(save_cif)
    sample_args.eval = True
    sample_args.eval_out_dir = scope_dir / "eval"
    sample_args.eval_stats_npz = npz
    sample_args.eval_min_dist = float(eval_min_dist)
    sample_args.eval_bond_cut = float(eval_bond_cut)
    sample_args.eval_dup_eps = float(eval_dup_eps)

    samples_path = sample_tokens_mod.run_sampling(sample_args)
    eval_dir = scope_dir / "eval"
    tier0_path = eval_dir / "tier0_metrics.json"
    tier1_path = eval_dir / "tier1_2d_metrics.json"
    return {
        "scope": scope_name,
        "cond_split": cond_split,
        "seed": int(seed),
        "paths": {
            "samples_npz": str(samples_path),
            "eval_dir": str(eval_dir),
            "tier0_metrics": str(tier0_path),
            "tier1_metrics": str(tier1_path),
            "per_sample": str(eval_dir / "per_sample.jsonl"),
        },
        "tier0": _load_json(tier0_path),
        "tier1": _load_json(tier1_path),
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    report: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": created_at,
        "run_metadata": collect_run_metadata(),
        "inputs": {
            "checkpoint": str(args.checkpoint),
            "npz": str(args.npz),
            "split_json": str(args.split_json),
        },
        "params": {
            "num_samples": int(args.num_samples),
            "steps": int(args.steps),
            "method": str(args.method),
            "seed": int(args.seed),
            "save_cif": bool(args.save_cif),
            "eval_min_dist": float(args.eval_min_dist),
            "eval_bond_cut": float(args.eval_bond_cut),
            "eval_dup_eps": float(args.eval_dup_eps),
        },
        "scopes": {},
    }

    report["scopes"]["condition_reconstruction_train"] = _run_one(
        scope_name="condition_reconstruction_train",
        cond_split="train",
        checkpoint=args.checkpoint,
        npz=args.npz,
        split_json=args.split_json,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        steps=args.steps,
        method=args.method,
        seed=args.seed,
        save_cif=args.save_cif,
        eval_min_dist=args.eval_min_dist,
        eval_bond_cut=args.eval_bond_cut,
        eval_dup_eps=args.eval_dup_eps,
    )
    report["scopes"]["conditional_generation_heldout"] = _run_one(
        scope_name="conditional_generation_heldout",
        cond_split="heldout",
        checkpoint=args.checkpoint,
        npz=args.npz,
        split_json=args.split_json,
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        steps=args.steps,
        method=args.method,
        seed=args.seed + 1,
        save_cif=args.save_cif,
        eval_min_dist=args.eval_min_dist,
        eval_bond_cut=args.eval_bond_cut,
        eval_dup_eps=args.eval_dup_eps,
    )

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

