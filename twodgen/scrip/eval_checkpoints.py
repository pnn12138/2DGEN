from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twodgen.scrip import sample_tokens as sample_tokens_mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate AtomDenoiser checkpoints via sampling + eval.")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory containing timestamped checkpoint subfolders.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        action="append",
        default=[],
        help="Basename(s) under checkpoints-dir to skip (can be repeated).",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Token cache NPZ used for volume bounds/clip and (optional) conditioning.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/eval_checkpoints"),
        help="Root output directory; each checkpoint writes to out-root/<run_name>/.",
    )
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--coord-frame", type=str, default="canon", choices=["raw", "canon"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--project-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--project-each-step", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-min-dist", type=float, default=1.5)
    parser.add_argument("--eval-bond-cut", type=float, default=3.0)
    parser.add_argument("--eval-dup-eps", type=float, default=1e-3)
    parser.add_argument("--eval-pbc-mask", type=str, default=None)
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="atomdenoiser_best.pt",
        help="Checkpoint filename inside each subfolder.",
    )
    return parser.parse_args(argv)


def _summarize_eval(out_dir: Path) -> dict[str, Any]:
    eval_dir = out_dir / "eval"
    tier0_path = eval_dir / "tier0_metrics.json"
    tier1_path = eval_dir / "tier1_2d_metrics.json"
    summary: dict[str, Any] = {"out_dir": str(out_dir)}
    if tier0_path.exists():
        tier0 = json.loads(tier0_path.read_text())
        summary["valid_rate_eval"] = tier0.get("valid_rate_eval")
        summary["fail_reason_counts"] = tier0.get("fail_reason_counts")
        summary["min_dist_mean"] = (tier0.get("min_dist") or {}).get("mean")
        summary["volume_mean"] = (tier0.get("volume") or {}).get("mean")
    if tier1_path.exists():
        tier1 = json.loads(tier1_path.read_text())
        summary["valid_2d_rate"] = tier1.get("valid_2d_rate")
        summary["vacuum_mean"] = (tier1.get("vacuum") or {}).get("mean")
        summary["thickness_mean"] = (tier1.get("thickness") or {}).get("mean")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if not args.checkpoints_dir.exists():
        raise FileNotFoundError(args.checkpoints_dir)

    runs = sorted([p for p in args.checkpoints_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    results: list[dict[str, Any]] = []

    for run_dir in runs:
        name = run_dir.name
        if name in set(args.exclude):
            continue
        ckpt_path = run_dir / args.checkpoint_name
        if not ckpt_path.exists():
            print(f"[skip] {name}: missing {args.checkpoint_name}")
            continue

        out_dir = args.out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd_args = [
            "--seed",
            str(args.seed),
            "--checkpoint",
            str(ckpt_path),
            "--npz",
            str(args.npz),
            "--num-samples",
            str(args.num_samples),
            "--steps",
            str(args.steps),
            "--method",
            args.method,
            "--out-dir",
            str(out_dir),
            "--coord-frame",
            args.coord_frame,
            "--no-save-cif",
            "--eval",
            "--eval-stats-npz",
            str(args.npz),
            "--eval-min-dist",
            str(args.eval_min_dist),
            "--eval-bond-cut",
            str(args.eval_bond_cut),
            "--eval-dup-eps",
            str(args.eval_dup_eps),
        ]
        if args.use_ema:
            cmd_args.append("--use-ema")
        if args.project_geometry:
            cmd_args.append("--project-geometry")
        else:
            cmd_args.append("--no-project-geometry")
        if args.project_each_step:
            cmd_args.append("--project-each-step")
        else:
            cmd_args.append("--no-project-each-step")
        if args.eval_pbc_mask is not None:
            cmd_args += ["--eval-pbc-mask", args.eval_pbc_mask]

        print(f"[run] {name}: sampling {args.num_samples} x {args.steps} ({args.method}) -> {out_dir}")
        sample_args = sample_tokens_mod.parse_args(cmd_args)
        sample_tokens_mod.run_sampling(sample_args)
        results.append({"run": name, **_summarize_eval(out_dir)})

    out_path = args.out_root / "summary.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()

