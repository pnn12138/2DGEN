from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter

from twodgen.scrip.sample_tokens import _expand_and_center_vacuum, relax_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for CHGNet relax_batch.")
    parser.add_argument("--samples", type=Path, required=True, help="samples.npz to pick bad samples from.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--relax-steps", type=int, default=50)
    parser.add_argument("--relax-fmax", type=float, default=0.05)
    parser.add_argument("--relax-vacuum", type=float, default=20.0)
    parser.add_argument("--relax-device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = np.load(args.samples)
    frac = samples["frac"]
    lattice = samples["lattice"]
    z = samples["z"]
    mask = samples.get("atom_mask")
    min_dist_pre = samples.get("min_dist_pre")
    if min_dist_pre is None:
        raise ValueError("samples.npz missing min_dist_pre; need baseline samples with collision stats.")

    worst_idx = np.argsort(min_dist_pre)[: int(args.num)].tolist()
    structures: List[Structure] = []
    for i in worst_idx:
        z_i = z[i]
        mask_i = mask[i] if mask is not None else np.ones_like(z_i, dtype=float)
        valid_idx = np.where((mask_i > 0.5) & (z_i > 0))[0]
        zs = [int(z_i[idx]) for idx in valid_idx]
        coords = frac[i][valid_idx]
        lattice_mat, coords, _ = _expand_and_center_vacuum(
            lattice[i], coords, float(args.relax_vacuum)
        )
        elements = [Element.from_Z(v) for v in zs]
        structure = Structure(lattice=lattice_mat, species=elements, coords=coords, coords_are_cartesian=False)
        structures.append(structure)

    relaxed_structures, energies, flags = relax_batch(
        structures, steps=int(args.relax_steps), fmax=float(args.relax_fmax), device=args.relax_device
    )
    for out_idx, (src_idx, struct, energy, ok) in enumerate(
        zip(worst_idx, relaxed_structures, energies, flags)
    ):
        writer = CifWriter(struct)
        writer.write_file(args.out_dir / f"relaxed_{out_idx}_src{src_idx}.cif")
        if energy is not None:
            (args.out_dir / f"relaxed_{out_idx}_energy.txt").write_text(str(energy))
        (args.out_dir / f"relaxed_{out_idx}_ok.txt").write_text(str(bool(ok)))

    print(f"Saved {len(relaxed_structures)} relaxed CIFs to {args.out_dir}")


if __name__ == "__main__":
    main()
