from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _lattice_to_gram6_row(lattice: np.ndarray) -> np.ndarray:
    """
    Compute Gram6 from a row-vector lattice (cart = frac @ lattice).

    lattice: (..., 3, 3)
    returns: (..., 6) as [G11,G22,G33,G12,G13,G23]
    """
    if lattice.shape[-2:] != (3, 3):
        raise ValueError(f"Expected lattice shape (...,3,3), got {lattice.shape}")
    gram = lattice @ np.swapaxes(lattice, -1, -2)
    return np.stack(
        [gram[..., 0, 0], gram[..., 1, 1], gram[..., 2, 2], gram[..., 0, 1], gram[..., 0, 2], gram[..., 1, 2]],
        axis=-1,
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate token cache NPZ to row-lattice Gram6 convention (G = lattice @ lattice^T)."
    )
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="Input legacy .npz")
    parser.add_argument("--out", dest="out_path", type=Path, default=None, help="Output .npz (default: in-place)")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file in-place (only if --out is not provided).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path: Path = args.in_path
    out_path: Path | None = args.out_path

    if out_path is None:
        if not args.in_place:
            raise SystemExit("Refusing to overwrite without `--in-place` (or provide `--out`).")
        out_path = in_path

    data = np.load(in_path, allow_pickle=False)
    if "lattice" not in data.files:
        raise SystemExit("NPZ missing `lattice`; cannot migrate without original lattice matrices.")

    lattice = data["lattice"].astype(np.float32)
    g_scale = float(data["g_scale"]) if "g_scale" in data.files else 1.0

    gram6_phys = _lattice_to_gram6_row(lattice)
    gram6 = (gram6_phys / g_scale).astype(np.float32)

    payload: Dict[str, Any] = {k: data[k] for k in data.files}
    payload["gram6"] = gram6
    payload["gram6_convention"] = np.array("row_lattice")
    payload["gram6_version"] = np.asarray(2, dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Migrated gram6 convention -> row_lattice: {in_path} -> {out_path}")


if __name__ == "__main__":
    main()

