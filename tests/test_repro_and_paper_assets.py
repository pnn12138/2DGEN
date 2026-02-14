from __future__ import annotations

import json
from pathlib import Path

from twodgen.evaluate.export_paper_assets import main as export_assets_main
from twodgen.evaluate.repro_manifest import main as repro_main


def test_repro_manifest_and_paper_assets(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "experiments.yaml"
    registry.write_text(
        "\n".join(
            [
                "experiments:",
                "  E1_1:",
                "    config: twodgen/configs/bench/E1_1.yaml",
                "    protocol: quick",
                "    paper_assets: [Fig2, Table1]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "E1_1" / "run_a"
    (run_dir / "samples").mkdir(parents=True)
    (run_dir / "samples" / "samples.npz").write_bytes(b"npz")
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "metrics_summary_v1",
                "git_commit": "abc",
                "timestamp": "2026-02-10T00:00:00+00:00",
                "experiment_id": "E1_1",
                "config_hash": "h",
                "seed": 0,
                "protocol": "quick",
                "success_geom_rate": 0.5,
                "valid_rate_eval": 0.4,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "failure_breakdown.json").write_text(
        json.dumps(
            {
                "schema_version": "failure_breakdown_v1",
                "git_commit": "abc",
                "timestamp": "2026-02-10T00:00:00+00:00",
                "experiment_id": "E1_1",
                "config_hash": "h",
                "seed": 0,
                "protocol": "quick",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "projection_stats.json").write_text(
        json.dumps(
            {
                "schema_version": "projection_stats_v1",
                "git_commit": "abc",
                "timestamp": "2026-02-10T00:00:00+00:00",
                "experiment_id": "E1_1",
                "config_hash": "h",
                "seed": 0,
                "protocol": "quick",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "run_metadata_v1",
                "git_commit": "abc",
                "timestamp": "2026-02-10T00:00:00+00:00",
                "experiment_id": "E1_1",
                "config_hash": "h",
                "seed": 0,
                "protocol": "quick",
                "run_metadata": {
                    "dependencies": {"python": "3.12", "torch": "2.5.1"},
                    "runtime": {"device": "cpu", "dtype": "float32", "cuda_version": None},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "STATUS.json").write_text(
        json.dumps(
            {
                "schema_version": "run_layout_v1",
                "git_commit": "abc",
                "timestamp": "2026-02-10T00:00:00+00:00",
                "experiment_id": "E1_1",
                "config_hash": "h",
                "seed": 0,
                "protocol": "quick",
                "status": "success",
            }
        ),
        encoding="utf-8",
    )
    agg = runs_root / "E1_1" / "_aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "summary.json").write_text(
        json.dumps({"variants": {"full_projection": {"success_geom_rate": {"mean": 0.5}, "valid_rate_eval": {"mean": 0.4}}}}),
        encoding="utf-8",
    )

    manifest_out = tmp_path / "repro_manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "repro_manifest",
            "--runs-root",
            str(runs_root),
            "--registry",
            str(registry),
            "--out",
            str(manifest_out),
        ],
    )
    repro_main()
    data = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert data["total_runs"] == 1

    assets_dir = tmp_path / "paper_assets"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_paper_assets",
            "--runs-root",
            str(runs_root),
            "--registry",
            str(registry),
            "--out-dir",
            str(assets_dir),
        ],
    )
    export_assets_main()
    assert (assets_dir / "tables" / "Table1.csv").exists()
    assert (assets_dir / "figures" / "Fig2.png").exists()

