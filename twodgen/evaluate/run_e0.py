from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional

from twodgen.common.run_metadata import collect_run_metadata
from twodgen.evaluate.protocol import get_protocol
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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_or_fill_projection_stats(
    *,
    src: Path,
    dst: Path,
    experiment_id: str,
    seed: int,
    protocol: str,
    cfg_hash: str,
    run_meta: dict,
) -> None:
    if src.exists():
        atomic_write_json(dst, _load_json(src))
        return
    payload = make_schema_payload(
        schema_version=PROJECTION_STATS_SCHEMA_VERSION,
        payload={"available": False, "reason": "not_emitted_by_sampling"},
        experiment_id=experiment_id,
        seed=seed,
        protocol=protocol,
        config_hash_value=cfg_hash,
        run_metadata=run_meta,
    )
    atomic_write_json(dst, payload)


def _copy_or_fill_run_metadata(
    *,
    src: Path,
    dst: Path,
    experiment_id: str,
    seed: int,
    protocol: str,
    cfg_hash: str,
    run_meta: dict,
) -> None:
    if src.exists():
        atomic_write_json(dst, _load_json(src))
        return
    payload = make_schema_payload(
        schema_version=RUN_METADATA_SCHEMA_VERSION,
        payload={"available": False, "reason": "not_emitted_by_sampling"},
        experiment_id=experiment_id,
        seed=seed,
        protocol=protocol,
        config_hash_value=cfg_hash,
        run_metadata=run_meta,
    )
    atomic_write_json(dst, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase0 E0 sanity experiment with status/resume.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment-id", type=str, default="E0")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", type=str, default="quick", choices=["quick", "final"])
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--sample-args",
        type=str,
        default="",
        help="Extra args forwarded to twodgen.scrip.sample_tokens.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proto = get_protocol(args.protocol)
    run_name = args.run_name
    if run_name is None:
        run_name = f"{args.protocol}_seed{args.seed}_n{args.num_samples}"
    paths = make_run_paths(
        experiment_id=args.experiment_id,
        runs_root=args.runs_root,
        run_name=run_name,
    )
    ensure_run_dirs(paths)

    run_meta = collect_run_metadata(argv=sys.argv)
    cfg_hash = config_hash(
        {
            "checkpoint": str(args.checkpoint),
            "npz": str(args.npz) if args.npz is not None else None,
            "seed": int(args.seed),
            "num_samples": int(args.num_samples),
            "steps": int(args.steps),
            "protocol": str(args.protocol),
            "protocol_default_num_samples": int(proto.num_samples),
            "protocol_default_seeds": list(proto.seeds),
            "sample_args": str(args.sample_args),
        }
    )

    if args.resume and paths.status.exists():
        try:
            status_payload = _load_json(paths.status)
            if status_payload.get("status") == "success":
                print(f"[info] {paths.run_dir} already success; skip due to --resume.")
                return
        except Exception:
            pass

    write_status(
        paths,
        status="running",
        experiment_id=args.experiment_id,
        seed=args.seed,
        protocol=args.protocol,
        config_hash_value=cfg_hash,
        note="phase0 E0 run started",
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
        str(args.seed),
        "--experiment-id",
        str(args.experiment_id),
        "--protocol",
        str(args.protocol),
    ]
    if args.npz is not None:
        cmd += ["--npz", str(args.npz)]
    if args.sample_args.strip():
        cmd += shlex.split(args.sample_args)

    try:
        subprocess.run(cmd, check=True)
        _copy_or_fill_run_metadata(
            src=paths.samples_dir / "run_metadata.json",
            dst=paths.run_metadata,
            experiment_id=args.experiment_id,
            seed=args.seed,
            protocol=args.protocol,
            cfg_hash=cfg_hash,
            run_meta=run_meta,
        )
        _copy_or_fill_projection_stats(
            src=paths.samples_dir / "projection_stats.json",
            dst=paths.projection_stats,
            experiment_id=args.experiment_id,
            seed=args.seed,
            protocol=args.protocol,
            cfg_hash=cfg_hash,
            run_meta=run_meta,
        )
        eval_dir = paths.samples_dir / "eval"
        if not eval_dir.exists():
            raise RuntimeError(f"missing eval output dir: {eval_dir}")
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
            raise RuntimeError("phase0 validation failed: " + "; ".join(errors))
        if paths.error_trace.exists():
            paths.error_trace.unlink()
        write_status(
            paths,
            status="success",
            experiment_id=args.experiment_id,
            seed=args.seed,
            protocol=args.protocol,
            config_hash_value=cfg_hash,
            note="phase0 E0 run finished",
            run_metadata=run_meta,
        )
        print(f"[ok] E0 completed: {paths.run_dir}")
    except Exception as exc:
        write_error_trace(paths, exc)
        write_status(
            paths,
            status="failed",
            experiment_id=args.experiment_id,
            seed=args.seed,
            protocol=args.protocol,
            config_hash_value=cfg_hash,
            note="phase0 E0 run failed",
            run_metadata=run_meta,
        )
        raise


if __name__ == "__main__":
    main()
