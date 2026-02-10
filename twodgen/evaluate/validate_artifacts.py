from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from twodgen.evaluate.run_layout import (
    FAILURE_BREAKDOWN_SCHEMA_VERSION,
    METRICS_SUMMARY_SCHEMA_VERSION,
    PROJECTION_STATS_SCHEMA_VERSION,
    RUN_LAYOUT_SCHEMA_VERSION,
    RUN_METADATA_SCHEMA_VERSION,
)


_REQUIRED_SCHEMA_FIELDS = (
    "schema_version",
    "git_commit",
    "timestamp",
    "experiment_id",
    "config_hash",
    "seed",
    "protocol",
)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_schema(path: Path, expected_version: str) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"missing file: {path}"
    data = _read_json(path)
    for key in _REQUIRED_SCHEMA_FIELDS:
        if key not in data:
            return False, f"{path.name} missing required field: {key}"
    if data.get("schema_version") != expected_version:
        return (
            False,
            f"{path.name} schema_version={data.get('schema_version')!r} != {expected_version!r}",
        )
    return True, ""


def validate_run_dir(run_dir: Path, *, require_success_status: bool = True) -> List[str]:
    errors: List[str] = []
    checks = [
        (run_dir / "run_metadata.json", RUN_METADATA_SCHEMA_VERSION),
        (run_dir / "projection_stats.json", PROJECTION_STATS_SCHEMA_VERSION),
        (run_dir / "metrics_summary.json", METRICS_SUMMARY_SCHEMA_VERSION),
        (run_dir / "failure_breakdown.json", FAILURE_BREAKDOWN_SCHEMA_VERSION),
        (run_dir / "STATUS.json", RUN_LAYOUT_SCHEMA_VERSION),
    ]
    for path, expected in checks:
        ok, msg = _check_schema(path, expected)
        if not ok:
            errors.append(msg)
    if require_success_status and (run_dir / "STATUS.json").exists():
        status = _read_json(run_dir / "STATUS.json").get("status")
        if status != "success":
            errors.append(f"STATUS.json status must be 'success', got {status!r}")
        if status == "success" and (run_dir / "error_trace.txt").exists():
            errors.append("error_trace.txt should not exist when STATUS is success")
    samples_npz = run_dir / "samples" / "samples.npz"
    if not samples_npz.exists():
        errors.append(f"missing file: {samples_npz}")
    tmp_files = sorted(run_dir.glob("*.tmp"))
    if tmp_files:
        errors.append(f"temporary files present: {[p.name for p in tmp_files]}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate phase0 run artifacts.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-require-success-status", action="store_true", default=False)
    args = parser.parse_args()
    errors = validate_run_dir(
        args.run_dir,
        require_success_status=not bool(args.no_require_success_status),
    )
    if errors:
        for e in errors:
            print(f"[error] {e}")
        raise SystemExit(1)
    print(f"[ok] phase0 artifacts valid: {args.run_dir}")


if __name__ == "__main__":
    main()
