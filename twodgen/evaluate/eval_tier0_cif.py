from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from twodgen.common.geometry_np import min_dist_and_shifts, thickness_vacuum


def _parse_pbc_mask(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must have three comma-separated values, e.g. 1,1,0")
    mask = tuple(int(p) for p in parts)
    if any(p not in (0, 1) for p in mask):
        raise ValueError("--pbc-mask values must be 0 or 1")
    return mask  # type: ignore[return-value]


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


def _read_cif_with_pymatgen(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from pymatgen.core import Structure
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("pymatgen is required to read CIF files.") from exc
    structure = Structure.from_file(str(path))
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    frac = np.asarray(structure.frac_coords, dtype=float)
    z = np.asarray(structure.atomic_numbers, dtype=int)
    return lattice, frac, z


def _read_cif(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _read_cif_with_pymatgen(path)
    except ImportError:
        try:
            from ase.io import read as ase_read  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Install pymatgen or ase to read CIF files.") from exc
        atoms = ase_read(str(path))
        lattice = np.asarray(atoms.cell.array, dtype=float)
        frac = np.asarray(atoms.get_scaled_positions(), dtype=float)
        z = np.asarray(atoms.numbers, dtype=int)
        return lattice, frac, z


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier-0 geometry checks from CIF inputs.")
    parser.add_argument("--cif-dir", type=Path, default=None, help="Directory with CIF files.")
    parser.add_argument("--cif-list", type=Path, default=None, help="Text file listing CIF paths.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--min-dist-cut", type=float, default=1.5)
    parser.add_argument("--bond-cut", type=float, default=3.0)
    parser.add_argument("--dup-eps", type=float, default=1e-3)
    parser.add_argument("--vacuum-min", type=float, default=None)
    parser.add_argument("--vacuum-ratio-min", type=float, default=None)
    parser.add_argument("--pbc-mask", type=str, default="1,1,0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pbc_mask = _parse_pbc_mask(args.pbc_mask)
    cif_paths = _load_cif_paths(args.cif_dir, args.cif_list)
    if not cif_paths:
        raise ValueError("No CIF files found.")

    per_sample: List[Dict[str, Any]] = []
    fail_counts: Dict[str, int] = {}
    min_dists: List[float] = []
    vacuums: List[float] = []
    vacuum_ratios: List[float] = []
    is_2d_flags: List[int] = []
    collision_flags: List[int] = []

    for idx, path in enumerate(cif_paths):
        lattice, frac, z = _read_cif(path)
        n_atoms = int(z.shape[0])
        reasons: List[str] = []

        if n_atoms < 3:
            reasons.append("low_atoms")

        if n_atoms > 0:
            min_dist, dist, _ = min_dist_and_shifts(frac, lattice, pbc_mask=pbc_mask)
            min_dists.append(min_dist)
            if min_dist < args.min_dist_cut:
                reasons.append("collision")
                collision_flags.append(1)
            else:
                collision_flags.append(0)
        else:
            min_dist = float("inf")
            dist = np.zeros((0, 0))
            min_dists.append(float("nan"))
            collision_flags.append(0)

        lengths = np.linalg.norm(lattice, axis=1)
        c_idx = int(np.argmax(lengths))
        c_len = float(lengths[c_idx])
        ab = [l for j, l in enumerate(lengths) if j != c_idx]
        vacuum_ratio = float(c_len / max(np.mean(ab), 1e-8))
        vacuum_ratios.append(vacuum_ratio)

        if n_atoms > 0:
            thickness, vacuum = thickness_vacuum(frac[:, c_idx], c_len)
        else:
            thickness, vacuum = float("nan"), float("nan")
        vacuums.append(vacuum)

        ok_vacuum = args.vacuum_min is None or (
            np.isfinite(vacuum) and float(vacuum) >= float(args.vacuum_min)
        )
        ok_ratio = args.vacuum_ratio_min is None or vacuum_ratio >= float(args.vacuum_ratio_min)
        is_2d_flag = int(ok_vacuum and ok_ratio)
        is_2d_flags.append(is_2d_flag)

        if args.vacuum_min is not None and not ok_vacuum:
            reasons.append("low_vacuum")
        if args.vacuum_ratio_min is not None and not ok_ratio:
            reasons.append("low_vacuum_ratio")

        valid = len(reasons) == 0
        for reason in reasons:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1

        per_sample.append(
            {
                "id": int(idx),
                "cif_path": str(path),
                "n_atoms": n_atoms,
                "min_dist": float(min_dist),
                "collision_flag": bool(min_dist < args.min_dist_cut) if np.isfinite(min_dist) else False,
                "vacuum": float(vacuum) if np.isfinite(vacuum) else float("nan"),
                "c_len": c_len,
                "vacuum_ratio": vacuum_ratio,
                "is_2d_flag": bool(is_2d_flag),
                "valid": bool(valid),
                "fail_reason": "+".join(reasons) if reasons else "",
            }
        )

    tier0 = {
        "total_samples": len(per_sample),
        "collision_rate": float(np.mean(collision_flags)) if collision_flags else 0.0,
        "min_dist": _summary_stats(min_dists),
        "vacuum": _summary_stats(vacuums),
        "vacuum_ratio": _summary_stats(vacuum_ratios),
        "is_2d_rate": float(np.mean(is_2d_flags)) if is_2d_flags else 0.0,
        "fail_reason_counts": fail_counts,
        "eval_params": {
            "min_dist_cut": float(args.min_dist_cut),
            "bond_cut": float(args.bond_cut),
            "dup_eps": float(args.dup_eps),
            "vacuum_min": args.vacuum_min,
            "vacuum_ratio_min": args.vacuum_ratio_min,
            "pbc_mask": pbc_mask,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.out_dir / "per_sample_tier0.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    (args.out_dir / "tier0_metrics.json").write_text(
        json.dumps(tier0, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"Saved Tier-0 CIF eval outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
