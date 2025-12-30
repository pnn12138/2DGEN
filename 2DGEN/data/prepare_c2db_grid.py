from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pymatgen.core import Structure

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data.torus import DEFAULT_TORUS_FREQS, torus_encode_np, torus_feature_dim  # noqa: E402


def row_to_grid(
    cif_str: str,
    max_atoms: int,
    atomic_scale: float,
    torus_freqs: tuple[int, ...],
    lattice_scale: float,
    angle_scale: float,
) -> Optional[np.ndarray]:
    """
    Convert a CIF string to a torus-encoded grid of shape (3, max_atoms, 3 * 2 * F):
    - channel 0: scaled atomic numbers, repeated across width
    - channel 1: torus-encoded fractional coordinates (sin/cos pairs)
    - channel 2: lattice parameters [a, b, c, alpha, beta, gamma] broadcast across atoms
      (angles divided by `angle_scale`, lengths divided by `lattice_scale`)
    """
    structure = Structure.from_str(cif_str, fmt="cif")
    num_atoms = len(structure)
    if num_atoms > max_atoms:
        return None

    atomic_numbers = np.asarray([site.specie.number for site in structure], dtype=np.float32)
    atomic_numbers = atomic_numbers / atomic_scale
    frac_coords = np.asarray(structure.frac_coords, dtype=np.float32)

    lengths = np.asarray(structure.lattice.abc, dtype=np.float32) / lattice_scale
    angles = np.asarray(structure.lattice.angles, dtype=np.float32) / angle_scale
    lattice_params = np.concatenate([lengths, angles], axis=0)

    frac_emb = torus_encode_np(frac_coords, freqs=torus_freqs)
    frac_dim = torus_feature_dim(torus_freqs)

    grid = np.zeros((3, max_atoms, frac_dim), dtype=np.float32)
    grid[0, :num_atoms, :] = atomic_numbers[:, None]
    grid[1, :num_atoms, :] = frac_emb
    grid[2, :, :6] = lattice_params[None, None, :]
    return grid


def build_dataset(
    csv_path: Path,
    max_atoms: int,
    atomic_scale: float,
    limit: Optional[int],
    torus_freqs: tuple[int, ...],
    lattice_scale: float,
    angle_scale: float,
) -> tuple[np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    grids: List[np.ndarray] = []
    material_ids: List[str] = []

    for row in df.itertuples(index=False):
        cif = getattr(row, "cif", None)
        if not isinstance(cif, str) or not cif.strip():
            continue
        try:
            grid = row_to_grid(
                cif,
                max_atoms=max_atoms,
                atomic_scale=atomic_scale,
                torus_freqs=torus_freqs,
                lattice_scale=lattice_scale,
                angle_scale=angle_scale,
            )
        except Exception:
            continue
        if grid is None:
            continue
        grids.append(grid)
        material_ids.append(str(getattr(row, "material_id", "")))

    if not grids:
        frac_dim = torus_feature_dim(torus_freqs)
        return np.zeros((0, 3, max_atoms, frac_dim), dtype=np.float32), []

    x = np.stack(grids, axis=0)
    return x, material_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert C2DB CSV to torus-encoded grids and save npz.")
    parser.add_argument("--csv", type=Path, default=Path("data/C2DB/c2db_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/C2DB/ache/c2db_grid.npz"))
    parser.add_argument("--max-atoms", type=int, default=24)
    parser.add_argument("--atomic-scale", type=float, default=100.0, help="Divide atomic numbers by this value.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for quick runs.")
    parser.add_argument(
        "--lattice-scale",
        type=float,
        default=10.0,
        help="Divide lattice constants a/b/c by this value before writing to grids.",
    )
    parser.add_argument(
        "--angle-scale",
        type=float,
        default=180.0,
        help="Divide lattice angles (deg) by this value before writing to grids.",
    )
    parser.add_argument(
        "--torus-freqs",
        type=int,
        nargs="+",
        default=DEFAULT_TORUS_FREQS,
        help="Frequencies used for torus sin/cos encoding of fractional coordinates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torus_freqs = tuple(args.torus_freqs)
    x, material_ids = build_dataset(
        csv_path=args.csv,
        max_atoms=args.max_atoms,
        atomic_scale=args.atomic_scale,
        limit=args.limit,
        torus_freqs=torus_freqs,
        lattice_scale=args.lattice_scale,
        angle_scale=args.angle_scale,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=x,
        material_id=np.array(material_ids),
        torus_freqs=np.asarray(torus_freqs, dtype=np.int64),
        lattice_scale=np.asarray(args.lattice_scale, dtype=np.float32),
        angle_scale=np.asarray(args.angle_scale, dtype=np.float32),
    )
    print(f"Saved {len(x)} samples to {args.out}")


if __name__ == "__main__":
    main()
