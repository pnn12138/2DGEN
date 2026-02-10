from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from twodgen.common.run_metadata import collect_run_metadata
from twodgen.evaluate.run_layout import (
    PROJECTION_STATS_SCHEMA_VERSION,
    RUN_METADATA_SCHEMA_VERSION,
    atomic_write_json,
    config_hash,
    ensure_run_dirs,
    make_run_paths,
    make_schema_payload,
    write_error_trace,
    write_status,
)
from twodgen.evaluate.validate_artifacts import validate_run_dir


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_or_fill_json(
    *,
    src: Path,
    dst: Path,
    schema_version: str,
    experiment_id: str,
    seed: int,
    protocol: str,
    cfg_hash: str,
    run_meta: Dict[str, Any],
    empty_reason: str,
) -> None:
    if src.exists():
        atomic_write_json(dst, _read_json(src))
        return
    payload = make_schema_payload(
        schema_version=schema_version,
        payload={"available": False, "reason": empty_reason},
        experiment_id=experiment_id,
        seed=seed,
        protocol=protocol,
        config_hash_value=cfg_hash,
        run_metadata=run_meta,
    )
    atomic_write_json(dst, payload)


def _variant_args(variant: str, cond_max: float) -> List[str]:
    common = ["--project-gram-cond", "--project-gram-max-cond", str(cond_max)]
    if variant == "baseline":
        return common + ["--project-final", "--no-post-project"]
    if variant == "full_projection":
        return common + [
            "--project-final",
            "--post-project",
            "--post-project-interval",
            "1",
            "--post-project-keys",
            "angle,cond,inplane,volume",
            "--post-project-cond-max",
            str(cond_max),
        ]
    if variant == "cond_only":
        return common + ["--project-final", "--post-project", "--post-project-interval", "1", "--post-project-keys", "cond"]
    if variant == "angle_only":
        return common + ["--project-final", "--post-project", "--post-project-interval", "1", "--post-project-keys", "angle"]
    if variant == "volume_only":
        return common + ["--project-final", "--post-project", "--post-project-interval", "1", "--post-project-keys", "volume"]
    if variant == "cond_angle":
        return common + [
            "--project-final",
            "--post-project",
            "--post-project-interval",
            "1",
            "--post-project-keys",
            "cond,angle",
            "--post-project-cond-max",
            str(cond_max),
        ]
    raise ValueError(f"Unknown variant: {variant}")


def _mean_std(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0.0}
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": float(mean_v), "std": float(std_v), "n": float(len(values))}


def _aggregate_by_variant(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)
    metrics = [
        "success_geom_rate",
        "bad_volume_rate",
        "collision_rate",
        "inplane_degen_rate",
        "cross_vacuum_risk_rate",
        "post_project_trigger_any_rate",
        "valid_rate_eval",
    ]
    out: Dict[str, Any] = {"variants": {}}
    for variant, items in grouped.items():
        agg: Dict[str, Any] = {"runs": len(items)}
        for key in metrics:
            vals: List[float] = []
            for it in items:
                value = it.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    vals.append(float(value))
            agg[key] = _mean_std(vals)
        out["variants"][variant] = agg
    if "baseline" in out["variants"] and "full_projection" in out["variants"]:
        b = out["variants"]["baseline"]["success_geom_rate"]["mean"]
        p = out["variants"]["full_projection"]["success_geom_rate"]["mean"]
        if math.isfinite(b) and math.isfinite(p):
            out["delta_success_geom_rate_full_minus_baseline"] = p - b
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E1 ablations (variant x seed) with phase0 artifacts.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--variants", type=str, required=True, help="Comma separated variants.")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--protocol", type=str, default="quick", choices=["quick", "final"])
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cond-max", type=float, default=40.0)
    parser.add_argument("--sample-args", type=str, default="")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-delta", type=float, default=None, help="Optional threshold for full-baseline success_geom delta.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_meta = collect_run_metadata(argv=sys.argv)
    rows: List[Dict[str, Any]] = []

    for variant in variants:
        for seed in seeds:
            run_name = f"{variant}_seed{seed}_n{args.num_samples}"
            paths = make_run_paths(
                experiment_id=args.experiment_id,
                runs_root=args.runs_root,
                run_name=run_name,
            )
            ensure_run_dirs(paths)
            cfg_hash = config_hash(
                {
                    "checkpoint": str(args.checkpoint),
                    "npz": str(args.npz) if args.npz is not None else None,
                    "protocol": str(args.protocol),
                    "num_samples": int(args.num_samples),
                    "steps": int(args.steps),
                    "seed": seed,
                    "variant": variant,
                    "cond_max": float(args.cond_max),
                    "sample_args": str(args.sample_args),
                }
            )
            if args.resume and paths.status.exists():
                try:
                    if _read_json(paths.status).get("status") == "success":
                        print(f"[info] skip success run: {paths.run_dir}")
                        metrics_path = paths.run_dir / "metrics_summary.json"
                        if metrics_path.exists():
                            row = _read_json(metrics_path)
                            row["variant"] = variant
                            row["seed"] = seed
                            row["run_dir"] = str(paths.run_dir)
                            rows.append(row)
                        continue
                except Exception:
                    pass

            write_status(
                paths,
                status="running",
                experiment_id=args.experiment_id,
                seed=seed,
                protocol=args.protocol,
                config_hash_value=cfg_hash,
                note=f"{variant} started",
                run_metadata=run_meta,
            )
            cmd = [
                sys.executable,
                "-m",
                "twodgen.scrip.sample_tokens",
                "--checkpoint",
                str(args.checkpoint),
                "--out-dir",
                str(paths.samples_dir),
                "--num-samples",
                str(args.num_samples),
                "--steps",
                str(args.steps),
                "--seed",
                str(seed),
                "--experiment-id",
                str(args.experiment_id),
                "--protocol",
                str(args.protocol),
            ]
            if args.npz is not None:
                cmd += ["--npz", str(args.npz)]
            cmd += _variant_args(variant, cond_max=float(args.cond_max))
            if args.sample_args.strip():
                cmd += shlex.split(args.sample_args)
            try:
                subprocess.run(cmd, check=True)
                _copy_or_fill_json(
                    src=paths.samples_dir / "run_metadata.json",
                    dst=paths.run_metadata,
                    schema_version=RUN_METADATA_SCHEMA_VERSION,
                    experiment_id=args.experiment_id,
                    seed=seed,
                    protocol=args.protocol,
                    cfg_hash=cfg_hash,
                    run_meta=run_meta,
                    empty_reason="not_emitted_by_sampling",
                )
                _copy_or_fill_json(
                    src=paths.samples_dir / "projection_stats.json",
                    dst=paths.projection_stats,
                    schema_version=PROJECTION_STATS_SCHEMA_VERSION,
                    experiment_id=args.experiment_id,
                    seed=seed,
                    protocol=args.protocol,
                    cfg_hash=cfg_hash,
                    run_meta=run_meta,
                    empty_reason="not_emitted_by_sampling",
                )
                eval_dir = paths.samples_dir / "eval"
                for name in (
                    "metrics_summary.json",
                    "failure_breakdown.json",
                    "tier0_metrics.json",
                    "tier1_2d_metrics.json",
                    "per_sample.jsonl",
                    "success_manifest.json",
                ):
                    src = eval_dir / name
                    if src.exists():
                        shutil.copy2(src, paths.run_dir / name)
                errors = validate_run_dir(paths.run_dir, require_success_status=False)
                if errors:
                    raise RuntimeError("; ".join(errors))
                if paths.error_trace.exists():
                    paths.error_trace.unlink()
                write_status(
                    paths,
                    status="success",
                    experiment_id=args.experiment_id,
                    seed=seed,
                    protocol=args.protocol,
                    config_hash_value=cfg_hash,
                    note=f"{variant} finished",
                    run_metadata=run_meta,
                )
            except Exception as exc:
                write_error_trace(paths, exc)
                write_status(
                    paths,
                    status="failed",
                    experiment_id=args.experiment_id,
                    seed=seed,
                    protocol=args.protocol,
                    config_hash_value=cfg_hash,
                    note=f"{variant} failed",
                    run_metadata=run_meta,
                )
                raise

            row = _read_json(paths.run_dir / "metrics_summary.json")
            row["variant"] = variant
            row["seed"] = seed
            row["run_dir"] = str(paths.run_dir)
            rows.append(row)

    aggregate = _aggregate_by_variant(rows)
    aggregate["experiment_id"] = args.experiment_id
    aggregate["protocol"] = args.protocol
    aggregate["num_samples"] = int(args.num_samples)
    aggregate["steps"] = int(args.steps)
    aggregate["variants_requested"] = variants
    aggregate["seeds_requested"] = seeds

    out_dir = args.runs_root / args.experiment_id / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "summary.json", aggregate)
    csv = ["variant,metric,mean,std,n"]
    for variant, stats in aggregate.get("variants", {}).items():
        for metric, v in stats.items():
            if not isinstance(v, dict) or "mean" not in v:
                continue
            csv.append(f"{variant},{metric},{v['mean']},{v['std']},{v['n']}")
    (out_dir / "summary.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")

    delta = aggregate.get("delta_success_geom_rate_full_minus_baseline")
    if args.require_delta is not None and delta is not None and float(delta) < float(args.require_delta):
        raise SystemExit(
            f"[fail] success_geom delta {delta:.4f} < required {args.require_delta:.4f}"
        )
    print(f"[ok] ablation completed: {out_dir}")


if __name__ == "__main__":
    main()
