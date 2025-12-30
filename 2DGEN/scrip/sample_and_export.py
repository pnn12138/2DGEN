"""
Legacy sampler for torus-encoded 3x24x24 grids (with lattice params in channel 2).
Token-based diffusion is the default path; see 2DGEN/scrip/sample_tokens.py.

Example:
    UV_CACHE_DIR=/home/pnn/2dgen/.uv_cache uv run python 2DGEN/scrip/sample_and_export.py \
        --checkpoint outputs/checkpoints/c2dbdenoiser_epoch1.pt \
        --num-samples 10 --steps 20 --out-dir outputs/samples_cif --lattice-scale 10 --angle-scale 180
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
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data.torus import DEFAULT_TORUS_FREQS, torus_decode_np, torus_feature_dim  # noqa: E402
from model.denoiser import C2DBDenoiser, DenoiserConfig  # noqa: E402
from model.model import JiTC2DBConfig  # noqa: E402


def prune_padding_rows(grid: np.ndarray, target_atoms: int, atom_threshold: float, large_neg: float = -1e6) -> np.ndarray:
    """
    SCDM-style hard masking: keep at most `target_atoms` rows with strongest atomic signal,
    drop the rest by adding a large negative bias so downstream logic filters them out.

    Args:
        grid: Sampled grid of shape (3, max_atoms, width).
        target_atoms: Maximum number of atoms to keep after pruning.
        atom_threshold: Minimum mean atomic signal to keep a row.
        large_neg: Value written into dropped rows to make atomic_numbers <= 0.
    """
    scores = grid[0].mean(axis=1)  # atomic channel strength per row
    keep_idx = np.argsort(scores)[::-1][:target_atoms]

    mask = np.zeros_like(scores, dtype=bool)
    mask[keep_idx] = scores[keep_idx] > atom_threshold

    pruned = grid.copy()
    pruned[0, ~mask, :] = large_neg
    pruned[1, ~mask, :] = 0.0  # optional: wipe coords to avoid accidental decoding
    # lattice channel kept as-is; it is read from row 0 during decoding
    return pruned


def grid_to_structure(
    grid: np.ndarray,
    atomic_scale: float,
    torus_freqs: tuple[int, ...],
    max_atoms: int,
    lattice_scale: float,
    angle_scale: float,
) -> Structure:
    """
    Convert a single torus-encoded grid back to a pymatgen Structure.

    Notes:
        - Atomic numbers were scaled by `atomic_scale` during preprocessing.
        - Lattice parameters are stored in channel 2, first 6 slots:
          [a, b, c, alpha, beta, gamma] divided by lattice_scale / angle_scale.
        - `max_atoms` should match the height used during preprocessing/model training.
    """
    if grid.shape[0] != 3:
        raise ValueError(f"Unexpected grid channels: {grid.shape[0]}")

    expected_width = torus_feature_dim(torus_freqs)
    if grid.shape[2] != expected_width:
        raise ValueError(f"Unexpected grid width: {grid.shape}, expected width {expected_width}")
    if grid.shape[1] != max_atoms:
        raise ValueError(f"Unexpected atom dimension: {grid.shape}, expected height {max_atoms}")

    atomic_vals = grid[0].mean(axis=1)
    frac_coords = torus_decode_np(grid[1], freqs=torus_freqs)

    coord_mask = np.linalg.norm(frac_coords, axis=1) > 1e-4
    atom_mask = (atomic_vals > 0.0) | coord_mask
    if atom_mask.sum() == 0:
        raise ValueError("No atoms found after applying mask surrogate.")

    atomic_numbers = np.rint(atomic_vals * atomic_scale).astype(int)
    atomic_numbers = np.clip(atomic_numbers, 1, 118)

    lattice_params = grid[2, 0, :6].copy()
    a, b, c = np.clip(lattice_params[:3] * lattice_scale, 0.5, None)
    alpha, beta, gamma = np.clip(lattice_params[3:] * angle_scale, 30.0, 150.0)

    elements: List[Element] = []
    coords: List[List[float]] = []
    for z, mask_val, coord in zip(atomic_numbers, atom_mask, frac_coords):
        if not mask_val:
            continue
        elements.append(Element.from_Z(int(z)))
        coords.append(coord.tolist())

    lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
    return Structure(lattice=lattice, species=elements, coords=coords, coords_are_cartesian=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample C2DBDenoiser outputs and export to CIF.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20, help="Sampling steps (Euler/Heun).")
    parser.add_argument("--method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=24,
        help="Atom dimension (height) expected by the model; should match preprocessing.",
    )
    parser.add_argument(
        "--torus-freqs",
        type=int,
        nargs="+",
        default=DEFAULT_TORUS_FREQS,
        help="Frequencies used for torus sin/cos encoding of fractional coordinates.",
    )
    parser.add_argument(
        "--save-decoded-frac",
        action="store_true",
        help="Also store decoded fractional coordinates in the samples npz for inspection.",
    )
    parser.add_argument("--atomic-scale", type=float, default=100.0, help="Scale factor used in preprocessing.")
    parser.add_argument(
        "--lattice-scale",
        type=float,
        default=10.0,
        help="Scale factor used when writing lattice constants during preprocessing.",
    )
    parser.add_argument(
        "--angle-scale",
        type=float,
        default=180.0,
        help="Scale factor used when writing lattice angles during preprocessing.",
    )
    parser.add_argument(
        "--target-atoms",
        type=int,
        default=None,
        help="Max atoms to keep during decoding; defaults to --max-atoms.",
    )
    parser.add_argument(
        "--atom-threshold",
        type=float,
        default=0.1,
        help="Row kept only if mean atomic channel exceeds this threshold before top-k.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/samples_cif"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torus_freqs = tuple(args.torus_freqs)

    img_size = (args.max_atoms, torus_feature_dim(torus_freqs))
    model_cfg = JiTC2DBConfig(img_size=img_size)
    model = C2DBDenoiser(DenoiserConfig(model=model_cfg)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_atoms = args.max_atoms if args.target_atoms is None else args.target_atoms

    with torch.no_grad():
        samples = model.generate(batch_size=args.num_samples, steps=args.steps, method=args.method)

    np_samples = samples.cpu().numpy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "x": np_samples,
        "torus_freqs": np.asarray(torus_freqs, dtype=np.int64),
        "lattice_scale": np.asarray(args.lattice_scale, dtype=np.float32),
        "angle_scale": np.asarray(args.angle_scale, dtype=np.float32),
    }
    if args.save_decoded_frac:
        # Decode fractional coordinates for convenience; shape (N, max_atoms, 3)
        save_kwargs["frac_decoded"] = torus_decode_np(np_samples[:, 1], freqs=torus_freqs)
    np.savez_compressed(args.out_dir / "samples.npz", **save_kwargs)

    for idx, grid in enumerate(np_samples):
        pruned_grid = prune_padding_rows(
            grid,
            target_atoms=target_atoms,
            atom_threshold=args.atom_threshold,
        )
        try:
            structure = grid_to_structure(
                pruned_grid,
                atomic_scale=args.atomic_scale,
                torus_freqs=torus_freqs,
                max_atoms=args.max_atoms,
                lattice_scale=args.lattice_scale,
                angle_scale=args.angle_scale,
            )
        except Exception as exc:  # pragma: no cover - export safety
            print(f"[warn] skip sample {idx} due to error: {exc}")
            continue
        CifWriter(structure).write_file(args.out_dir / f"sample_{idx}.cif")

    print(f"Saved {len(np_samples)} grids to {args.out_dir} (npz + CIFs).")


if __name__ == "__main__":
    main()
