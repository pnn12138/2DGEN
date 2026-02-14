from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pymatgen.core import Element, Structure


DEFAULT_INCAR = """SYSTEM = twodgen_spotcheck
ENCUT = 520
EDIFF = 1E-5
ISMEAR = 0
SIGMA = 0.05
IBRION = 2
NSW = 100
ISIF = 3
PREC = Accurate
"""

DEFAULT_KPOINTS = """Automatic mesh
0
Gamma
6 6 1
0 0 0
"""


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _selected_rows(rows: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
    selected = [r for r in rows if str(r.get("selected_top_k", "0")) in ("1", "true", "True")]
    if not selected:
        selected = rows
    selected = sorted(selected, key=lambda r: int(float(r.get("rank_energy", "1e9"))))
    return selected[: max(int(k), 0)]


def _load_samples_path(rows: List[Dict[str, str]], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for row in rows:
        sp = row.get("samples_path")
        if sp:
            return Path(sp)
    raise ValueError("Cannot resolve samples path from screening.csv; pass --samples.")


def _write_poscar(samples: Dict[str, np.ndarray], sample_id: int, path: Path) -> List[str]:
    z = np.asarray(samples["z"][sample_id])
    frac = np.asarray(samples["frac"][sample_id])
    lattice = np.asarray(samples["lattice"][sample_id])
    mask = np.asarray(samples["atom_mask"][sample_id]) > 0.5
    z_valid = z[mask].astype(int)
    frac_valid = frac[mask]
    species = [Element.from_Z(int(val)).symbol for val in z_valid.tolist()]
    structure = Structure(lattice=lattice, species=species, coords=frac_valid, coords_are_cartesian=False)
    path.write_text(structure.to(fmt="poscar"), encoding="utf-8")
    return sorted(set(species))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-K candidates into DFT spot-check job folders.")
    parser.add_argument("--screening-csv", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--incar-template", type=Path, default=None)
    parser.add_argument("--kpoints-template", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_csv(args.screening_csv)
    pick = _selected_rows(rows, k=args.k)
    samples_path = _load_samples_path(rows, explicit=args.samples)
    samples = dict(np.load(samples_path))

    out_dir = args.out_dir
    jobs_dir = out_dir / "dft_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    incar_text = (
        args.incar_template.read_text(encoding="utf-8")
        if args.incar_template is not None
        else DEFAULT_INCAR
    )
    kpoints_text = (
        args.kpoints_template.read_text(encoding="utf-8")
        if args.kpoints_template is not None
        else DEFAULT_KPOINTS
    )

    manifest_rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(pick, start=1):
        sample_id = int(float(row["sample_id"]))
        job_id = f"job_{rank:04d}_sid_{sample_id:05d}"
        job_dir = jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        elems = _write_poscar(samples, sample_id, job_dir / "POSCAR")
        (job_dir / "INCAR").write_text(incar_text, encoding="utf-8")
        (job_dir / "KPOINTS").write_text(kpoints_text, encoding="utf-8")
        (job_dir / "POTCAR.ref").write_text(
            "# Placeholder only. Replace with site-specific POTCAR.\n" + " ".join(elems) + "\n",
            encoding="utf-8",
        )
        predicted = _float_or_none(row.get("formation_energy_per_atom"))
        if predicted is None:
            predicted = _float_or_none(row.get("energy_mlip"))
        job_meta = {
            "job_id": job_id,
            "sample_id": sample_id,
            "rank": rank,
            "predicted_energy": predicted,
            "screening_row": row,
        }
        (job_dir / "job.json").write_text(json.dumps(job_meta, indent=2, ensure_ascii=True), encoding="utf-8")
        manifest_rows.append(
            {
                "job_id": job_id,
                "sample_id": sample_id,
                "run_path": row.get("run_path", ""),
                "predicted_energy": predicted if predicted is not None else "",
                "rank": rank,
                "job_dir": str(job_dir),
            }
        )

    manifest_path = out_dir / "dft_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["job_id", "sample_id", "run_path", "predicted_energy", "rank", "job_dir"]
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)
    print(f"Saved DFT spot-check jobs to {jobs_dir}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()

