from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


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


def _read_cif(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, List[str]]:
    try:
        from pymatgen.core import Structure
    except ImportError:
        Structure = None  # type: ignore

    if Structure is not None:
        structure = Structure.from_file(str(path))
        lattice = np.asarray(structure.lattice.matrix, dtype=float)
        frac = np.asarray(structure.frac_coords, dtype=float)
        numbers = np.asarray(structure.atomic_numbers, dtype=int)
        formula = structure.composition.reduced_formula
        elements = sorted({str(el) for el in structure.composition.elements})
        return lattice, frac, numbers, formula, elements

    try:
        from ase.io import read as ase_read  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Install pymatgen or ase to read CIF files.") from exc
    atoms = ase_read(str(path))
    lattice = np.asarray(atoms.cell.array, dtype=float)
    frac = np.asarray(atoms.get_scaled_positions(), dtype=float)
    numbers = np.asarray(atoms.numbers, dtype=int)
    formula = atoms.get_chemical_formula(mode="hill")
    elements = sorted(set(atoms.get_chemical_symbols()))
    return lattice, frac, numbers, formula, elements


def _parse_target_elements(value: Optional[str]) -> Optional[Sequence[str]]:
    if value is None:
        return None
    elems = [v.strip() for v in value.split(",") if v.strip()]
    return elems if elems else None


def _match_formula(formula: str, target_formula: Optional[str]) -> bool:
    if target_formula is None:
        return True
    return formula == target_formula


def _match_elements(elements: Sequence[str], target_elements: Optional[Sequence[str]]) -> bool:
    if target_elements is None:
        return True
    return set(elements) == set(target_elements)


def _spacegroup_number(
    lattice: np.ndarray, frac: np.ndarray, numbers: np.ndarray
) -> Optional[int]:
    try:
        import spglib
    except ImportError:  # pragma: no cover - environment dependent
        return None
    cell = (lattice, frac, numbers)
    dataset = spglib.get_symmetry_dataset(cell)
    if dataset is None:
        return None
    return int(dataset.get("number"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check composition/symmetry conditions for CIFs.")
    parser.add_argument("--cif-dir", type=Path, default=None)
    parser.add_argument("--cif-list", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-formula", type=str, default=None)
    parser.add_argument("--target-elements", type=str, default=None, help="Comma-separated elements.")
    parser.add_argument("--target-spacegroup", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cif_paths = _load_cif_paths(args.cif_dir, args.cif_list)
    if not cif_paths:
        raise ValueError("No CIF files found.")

    target_elements = _parse_target_elements(args.target_elements)

    per_sample: List[Dict[str, Any]] = []
    fail_counts: Dict[str, int] = {}
    match_flags: List[int] = []

    for idx, path in enumerate(cif_paths):
        lattice, frac, numbers, formula, elements = _read_cif(path)
        sg_number = _spacegroup_number(lattice, frac, numbers)

        reasons: List[str] = []
        if not _match_formula(formula, args.target_formula):
            reasons.append("formula_mismatch")
        if not _match_elements(elements, target_elements):
            reasons.append("elements_mismatch")
        if args.target_spacegroup is not None:
            if sg_number is None or sg_number != args.target_spacegroup:
                reasons.append("spacegroup_mismatch")

        cond_match = len(reasons) == 0
        match_flags.append(int(cond_match))
        for reason in reasons:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1

        per_sample.append(
            {
                "id": int(idx),
                "cif_path": str(path),
                "formula": formula,
                "elements": elements,
                "spacegroup_number": sg_number,
                "cond_match": cond_match,
                "fail_reason": "+".join(reasons) if reasons else "",
            }
        )

    metrics = {
        "cond_match_rate": float(np.mean(match_flags)) if match_flags else 0.0,
        "fail_reason_counts": fail_counts,
        "total_samples": len(per_sample),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.out_dir / "per_sample_conditions.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    (args.out_dir / "conditions_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(f"Saved condition checks to {args.out_dir}")


if __name__ == "__main__":
    main()
