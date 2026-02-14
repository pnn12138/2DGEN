from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from twodgen.evaluate.check_mode_collapse import main as collapse_main
from twodgen.evaluate.diversity import main as diversity_main


def test_diversity_and_collapse_tools(tmp_path: Path, monkeypatch) -> None:
    per_sample = tmp_path / "per_sample.jsonl"
    rows = [
        {"id": 0, "n_atoms": 4, "inplane_area": 8.0, "spacegroup_number": 1},
        {"id": 1, "n_atoms": 8, "inplane_area": 16.0, "spacegroup_number": 2},
    ]
    per_sample.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n", encoding="utf-8")
    samples_npz = tmp_path / "samples.npz"
    train_npz = tmp_path / "train.npz"
    np.savez_compressed(
        samples_npz,
        z=np.array([[14, 14], [8, 14]], dtype=np.int64),
        atom_mask=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    )
    np.savez_compressed(
        train_npz,
        z=np.array([[14, 14], [14, 14]], dtype=np.int64),
        atom_mask=np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    )
    diversity_out = tmp_path / "diversity.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diversity",
            "--per-sample",
            str(per_sample),
            "--samples",
            str(samples_npz),
            "--train-npz",
            str(train_npz),
            "--out",
            str(diversity_out),
        ],
    )
    diversity_main()
    data = json.loads(diversity_out.read_text(encoding="utf-8"))
    assert data["composition_coverage"]["available"] is True

    baseline = tmp_path / "baseline_div.json"
    current = tmp_path / "current_div.json"
    baseline.write_text(
        json.dumps(
            {
                "spacegroup": {"coverage_vs_230": 0.5},
                "n_atoms_coverage": {"coverage": 0.5},
                "lattice_coverage": {"coverage": 0.5},
            }
        ),
        encoding="utf-8",
    )
    current.write_text(
        json.dumps(
            {
                "spacegroup": {"coverage_vs_230": 0.2},
                "n_atoms_coverage": {"coverage": 0.2},
                "lattice_coverage": {"coverage": 0.2},
            }
        ),
        encoding="utf-8",
    )
    collapse_out = tmp_path / "collapse_report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_mode_collapse",
            "--baseline-diversity",
            str(baseline),
            "--current-diversity",
            str(current),
            "--out",
            str(collapse_out),
        ],
    )
    collapse_main()
    report = json.loads(collapse_out.read_text(encoding="utf-8"))
    assert report["coverage_pass"] is False
    assert report["pass"] is False

