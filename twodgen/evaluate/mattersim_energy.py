from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _load_reference(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


def _formation_from_row(
    row: Dict[str, Any],
    ref: Dict[str, float],
    *,
    missing_strategy: str,
    default_mu: float,
) -> Tuple[Optional[float], Optional[str]]:
    comp = row.get("composition") or {}
    total_energy = row.get("total_energy")
    n_atoms = int(row.get("n_atoms") or 0)
    missing = [el for el in comp.keys() if el not in ref]
    if missing and missing_strategy == "fail":
        return None, "missing_ref_energy"
    if total_energy is None or n_atoms <= 0:
        return None, "invalid_input"
    ref_energy = 0.0
    for el, count in comp.items():
        mu = float(ref.get(el, default_mu))
        ref_energy += mu * int(count)
    formation = (float(total_energy) - ref_energy) / max(n_atoms, 1)
    return formation, None


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
    parser.add_argument("--ref-energies", type=Path, default=None, help="JSON mapping element->mu.")
    parser.add_argument("--formation-max", type=float, default=0.0)
    parser.add_argument(
        "--missing-strategy",
        type=str,
        choices=["fail", "default"],
        default="fail",
        help="How to handle missing element reference energies.",
    )
    parser.add_argument(
        "--default-mu",
        type=float,
        default=0.0,
        help="Fallback reference energy when missing-strategy=default.",
    )
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

    formation_written = False
    if args.ref_energies is not None:
        ref = _load_reference(args.ref_energies)
        per_sample_form: List[Dict[str, Any]] = []
        pass_flags: List[int] = []
        missing_ref: List[str] = []
        formation_values: List[float] = []
        for row in per_sample:
            comp = row.get("composition") or {}
            missing = [el for el in comp.keys() if el not in ref]
            if missing:
                missing_ref.extend(missing)
            formation, fail_reason = _formation_from_row(
                row,
                ref,
                missing_strategy=args.missing_strategy,
                default_mu=args.default_mu,
            )
            formation_pass = (
                formation is not None and formation <= float(args.formation_max)
            )
            pass_flags.append(int(formation_pass))
            if formation is not None:
                formation_values.append(float(formation))
            per_sample_form.append(
                {
                    "id": row.get("id"),
                    "cif_path": row.get("cif_path"),
                    "formation_energy_per_atom": formation,
                    "formation_pass": bool(formation_pass),
                    "threshold": float(args.formation_max),
                    "fail_reason": "" if fail_reason is None else fail_reason,
                }
            )

        metrics = {
            "formation_pass_rate": float(sum(pass_flags) / max(len(pass_flags), 1)),
            "formation_energy": {
                "count": len(formation_values),
                "mean": float(sum(formation_values) / max(len(formation_values), 1))
                if formation_values
                else None,
                "median": float(sorted(formation_values)[len(formation_values) // 2])
                if formation_values
                else None,
            },
            "missing_ref_elements": sorted(set(missing_ref)),
            "missing_strategy": args.missing_strategy,
            "default_mu": args.default_mu if args.missing_strategy == "default" else None,
        }

        formation_path = args.out_dir / "per_sample_formation.jsonl"
        with formation_path.open("w", encoding="utf-8") as f:
            for row in per_sample_form:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        (args.out_dir / "formation_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        formation_written = True

    if formation_written:
        print(f"Saved MatterSim energies and formation metrics to {args.out_dir}")
    else:
        print(f"Saved MatterSim energies to {out_path}")


if __name__ == "__main__":
    main()
