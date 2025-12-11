"""
Sample 3x24x3 grids from a trained C2DBDenoiser and export to CIF files.

Example:
    UV_CACHE_DIR=/home/pnn/2dgen/.uv_cache uv run python 2DGEN/sample_and_export.py \
        --checkpoint outputs/checkpoints/c2dbdenoiser_epoch1.pt \
        --num-samples 10 --steps 20 --out-dir outputs/samples_cif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifWriter

# Ensure the 2DGEN package directory is importable even though the folder starts with a digit.
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from model.denoiser import C2DBDenoiser  # noqa: E402


def grid_to_structure(
    grid: np.ndarray,
    atomic_scale: float,
    lattice_constant: float,
) -> Structure:
    """
    Convert a single (3, max_atoms, 3) grid back to a pymatgen Structure.

    Notes:
        - Atomic numbers were scaled by `atomic_scale` during preprocessing.
        - Lattice information is unavailable in the grid; we assume a cubic lattice with
          edge length `lattice_constant` Angstroms to materialize a valid CIF.
    """
    if grid.shape != (3, grid.shape[1], 3):
        raise ValueError(f"Unexpected grid shape: {grid.shape}")

    atom_mask = grid[2].mean(axis=1) > 0.5
    if atom_mask.sum() == 0:
        raise ValueError("No atoms found after applying mask.")

    atomic_vals = grid[0].mean(axis=1)
    atomic_numbers = np.clip(np.rint(atomic_vals * atomic_scale), 1, 118).astype(int)
    frac_coords = np.clip(grid[1], 0.0, 1.0)

    elements: List[Element] = []
    coords: List[List[float]] = []
    for z, mask_val, coord in zip(atomic_numbers, atom_mask, frac_coords):
        if not mask_val:
            continue
        elements.append(Element.from_Z(int(z)))
        coords.append(coord.tolist())

    lattice = Lattice.cubic(lattice_constant)
    return Structure(lattice=lattice, species=elements, coords=coords, coords_are_cartesian=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample C2DBDenoiser outputs and export to CIF.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps (Euler/Heun).")
    parser.add_argument("--method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--atomic-scale", type=float, default=100.0, help="Scale factor used in preprocessing.")
    parser.add_argument(
        "--lattice-constant",
        type=float,
        default=5.0,
        help="Edge length (Angstrom) for synthetic cubic lattice in CIF export.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/samples_cif"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = C2DBDenoiser().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        samples = model.generate(batch_size=args.num_samples, steps=args.steps, method=args.method)

    np_samples = samples.cpu().numpy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "samples.npz", x=np_samples)

    for idx, grid in enumerate(np_samples):
        try:
            structure = grid_to_structure(grid, atomic_scale=args.atomic_scale, lattice_constant=args.lattice_constant)
        except Exception as exc:  # pragma: no cover - export safety
            print(f"[warn] skip sample {idx} due to error: {exc}")
            continue
        CifWriter(structure).write_file(args.out_dir / f"sample_{idx}.cif")

    print(f"Saved {len(np_samples)} grids to {args.out_dir} (npz + CIFs).")


if __name__ == "__main__":
    main()
