from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_cif_paths(cif_dir: Optional[Path], cif_list: Optional[Path]) -> List[Path]:
    if cif_dir is None and cif_list is None:
        raise ValueError("Provide either --cif-dir or --cif-list.")
    if cif_dir is not None and cif_list is not None:
        raise ValueError("Use only one of --cif-dir or --cif-list.")
    if cif_dir is not None:
        return sorted([p for p in cif_dir.iterdir() if p.suffix.lower() == ".cif"])
    paths = []
    assert cif_list is not None
    for line in cif_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths


def get_atoms_from_cif(path: Path):
    try:
        from ase.io import read as ase_read  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("ase is required to read CIF files for MatterSim.") from exc
    atoms = ase_read(str(path))
    if len(atoms) == 0:
        raise ValueError("empty_atoms")
    if not atoms.cell.any():
        raise ValueError("missing_cell")
    return atoms


def _composition(atoms) -> Dict[str, int]:
    symbols = atoms.get_chemical_symbols()
    comp: Dict[str, int] = {}
    for sym in symbols:
        comp[sym] = comp.get(sym, 0) + 1
    return comp


def _load_mattersim(model_path: Optional[str], device: str):
    try:
        from mattersim.forcefield import MatterSimCalculator, Potential  # type: ignore
        from mattersim.applications.relax import BatchRelaxer  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("mattersim is required for energy evaluation.") from exc

    if model_path:
        potential = Potential.from_checkpoint(model_path, device=device)
    else:
        potential = Potential.from_checkpoint(device=device)
    calculator = MatterSimCalculator(potential=potential)
    relaxer = BatchRelaxer(potential, fmax=float(0.02), optimizer="FIRE", filter="ExpCellFilter")
    return calculator, relaxer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MatterSim energy for CIFs.")
    parser.add_argument("--cif-dir", type=Path, default=None)
    parser.add_argument("--cif-list", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=str, default=None, help="MatterSim checkpoint path.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--relax", action="store_true", help="Run quick relaxation.")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cif_paths = _load_cif_paths(args.cif_dir, args.cif_list)
    if not cif_paths:
        raise ValueError("No CIF files found.")

    calculator, relaxer = _load_mattersim(args.model_path, args.device)

    per_sample: List[Dict[str, Any]] = []
    atoms_list = []
    valid_paths = []
    for path in cif_paths:
        try:
            atoms = get_atoms_from_cif(path)
        except Exception as exc:
            per_sample.append(
                {
                    "id": len(per_sample),
                    "cif_path": str(path),
                    "n_atoms": 0,
                    "total_energy": None,
                    "energy_per_atom": None,
                    "relaxed": bool(args.relax),
                    "composition": {},
                    "status": "error",
                    "fail_reason": str(exc),
                }
            )
            continue
        atoms_list.append(atoms)
        valid_paths.append(path)

    if atoms_list and args.relax:
        trajectories = relaxer.relax(atoms_list, fmax=args.fmax, steps=args.steps)
        atoms_list = [traj[-1] for traj in trajectories]

    for idx, (path, atoms) in enumerate(zip(valid_paths, atoms_list)):
        atoms.calc = calculator
        total_energy = float(atoms.get_potential_energy())
        n_atoms = int(len(atoms))
        energy_per_atom = total_energy / max(n_atoms, 1)
        comp = _composition(atoms)
        per_sample.append(
            {
                "id": int(idx),
                "cif_path": str(path),
                "n_atoms": n_atoms,
                "total_energy": total_energy,
                "energy_per_atom": energy_per_atom,
                "relaxed": bool(args.relax),
                "composition": comp,
                "status": "ok",
                "fail_reason": "",
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "per_sample_energy.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Saved MatterSim energies to {out_path}")


if __name__ == "__main__":
    main()
