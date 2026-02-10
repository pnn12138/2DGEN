from __future__ import annotations

import json
from pathlib import Path

from twodgen.evaluate.aggregate_runs import main as aggregate_main
from twodgen.evaluate.eval_samples import write_eval_outputs
from twodgen.evaluate.io import load_eval_outputs
from twodgen.evaluate.protocol import get_protocol


def test_protocol_presets() -> None:
    quick = get_protocol("quick")
    final = get_protocol("final")
    assert quick.num_samples == 2000
    assert final.num_samples == 20000
    assert quick.seeds == [0, 1, 2]
    assert final.seeds == [0, 1, 2]


def test_eval_outputs_write_phase0_artifacts(tmp_path: Path) -> None:
    write_eval_outputs(
        out_dir=tmp_path,
        per_sample=[{"id": 0, "valid": True}],
        tier0={
            "success_geom_rate": 0.5,
            "success_rate": 0.4,
            "valid_rate_eval": 0.5,
            "min_dist_collision_rate": 0.1,
            "inplane_degen_rate": 0.05,
            "bad_volume_rate": 0.2,
            "total_samples": 1,
            "fail_reason_counts": {"collision": 1},
        },
        tier1={"cross_vacuum_rate": 0.2, "vacuum_ok_rate": 0.8},
        eval_params={"min_dist_cut": 1.5},
        run_context={"experiment_id": "E0", "seed": 0, "protocol": "quick", "config_hash": "abc"},
    )

    out = load_eval_outputs(tmp_path)
    assert out["metrics_summary"] is not None
    assert out["failure_breakdown"] is not None
    assert out["metrics_summary"]["schema_version"] == "metrics_summary_v1"
    assert out["failure_breakdown"]["schema_version"] == "failure_breakdown_v1"
    assert out["metrics_summary"]["experiment_id"] == "E0"
    assert out["failure_breakdown"]["seed"] == 0


def test_aggregate_runs_writes_summary(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "E1_1" / "20260210_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "metrics_summary_v1",
                "git_commit": "deadbeef",
                "timestamp": "2026-02-10T00:00:00+00:00",
                "experiment_id": "E1_1",
                "config_hash": "h",
                "seed": 0,
                "protocol": "quick",
                "success_geom_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["aggregate_runs", "--runs-root", str(runs_root), "--experiment-id", "E1_1"],
    )
    aggregate_main()
    summary_path = runs_root / "E1_1" / "_aggregate" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "metrics_summary_v1"
    assert "success_geom_rate" in summary["metrics"]

