from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from twodgen.evaluate.export_dft_spotcheck import main as export_main
from twodgen.evaluate.import_dft_results import main as import_main


def test_export_and_import_dft_spotcheck(tmp_path: Path, monkeypatch) -> None:
    samples_path = tmp_path / "samples.npz"
    np.savez_compressed(
        samples_path,
        z=np.array([[14, 0, 0]], dtype=np.int64),
        frac=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32),
        lattice=np.array([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]]], dtype=np.float32),
        atom_mask=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    screening_csv = tmp_path / "screening.csv"
    with screening_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "rank_energy",
                "selected_top_k",
                "formation_energy_per_atom",
                "energy_mlip",
                "success_geom",
                "energy_available",
                "run_path",
                "samples_path",
                "candidate_cif",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": 0,
                "rank_energy": 1,
                "selected_top_k": 1,
                "formation_energy_per_atom": -0.5,
                "energy_mlip": -5.0,
                "success_geom": 1,
                "energy_available": 1,
                "run_path": str(tmp_path),
                "samples_path": str(samples_path),
                "candidate_cif": "",
            }
        )

    out_dir = tmp_path / "dft_export"
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_dft_spotcheck",
            "--screening-csv",
            str(screening_csv),
            "--out-dir",
            str(out_dir),
            "--k",
            "1",
        ],
    )
    export_main()

    manifest = out_dir / "dft_manifest.csv"
    assert manifest.exists()
    job_dir = next((out_dir / "dft_jobs").iterdir())
    (job_dir / "energy.txt").write_text("-5.123\n", encoding="utf-8")

    out_summary = tmp_path / "import_summary.json"
    out_screening = tmp_path / "screening_with_dft.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "import_dft_results",
            "--manifest",
            str(manifest),
            "--screening-csv",
            str(screening_csv),
            "--out-screening",
            str(out_screening),
            "--out-summary",
            str(out_summary),
        ],
    )
    import_main()
    summary = json.loads(out_summary.read_text(encoding="utf-8"))
    assert summary["dft_available_rows"] == 1
    assert out_screening.exists()

