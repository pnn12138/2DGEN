from __future__ import annotations

import json
from pathlib import Path

from twodgen.evaluate.run_layout import (
    FAILURE_BREAKDOWN_SCHEMA_VERSION,
    METRICS_SUMMARY_SCHEMA_VERSION,
    PROJECTION_STATS_SCHEMA_VERSION,
    RUN_LAYOUT_SCHEMA_VERSION,
    RUN_METADATA_SCHEMA_VERSION,
)
from twodgen.evaluate.validate_artifacts import validate_run_dir


def _base_payload(schema_version: str) -> dict:
    return {
        "schema_version": schema_version,
        "git_commit": "deadbeef",
        "timestamp": "2026-02-10T00:00:00+00:00",
        "experiment_id": "E0",
        "config_hash": "hash",
        "seed": 0,
        "protocol": "quick",
    }


def test_validate_run_dir_ok(tmp_path: Path) -> None:
    run_dir = tmp_path
    (run_dir / "samples").mkdir()
    (run_dir / "samples" / "samples.npz").write_bytes(b"npz")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(_base_payload(RUN_METADATA_SCHEMA_VERSION)), encoding="utf-8"
    )
    (run_dir / "projection_stats.json").write_text(
        json.dumps(_base_payload(PROJECTION_STATS_SCHEMA_VERSION)), encoding="utf-8"
    )
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(_base_payload(METRICS_SUMMARY_SCHEMA_VERSION)), encoding="utf-8"
    )
    (run_dir / "failure_breakdown.json").write_text(
        json.dumps(_base_payload(FAILURE_BREAKDOWN_SCHEMA_VERSION)), encoding="utf-8"
    )
    status = _base_payload(RUN_LAYOUT_SCHEMA_VERSION)
    status["status"] = "success"
    (run_dir / "STATUS.json").write_text(json.dumps(status), encoding="utf-8")
    assert validate_run_dir(run_dir) == []


def test_validate_run_dir_missing_file(tmp_path: Path) -> None:
    errors = validate_run_dir(tmp_path, require_success_status=False)
    assert any("run_metadata.json" in e for e in errors)

