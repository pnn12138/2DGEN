from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from common.crystal import frac_mic_dist  # noqa: E402
from model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig  # noqa: E402
from common.atom_diffusion import AtomDiffusionConfig  # noqa: E402
from model.atom_transformer import AtomTransformerConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample token-based crystal diffusion and export CIF.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npz", type=Path, default=None, help="Token cache for sampling N/volume stats.")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--method", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--max-atoms", type=int, default=24)
    parser.add_argument("--num-atoms", type=int, default=None, help="Number of atoms to sample (<= max-atoms).")
    parser.add_argument("--g-scale", type=float, default=100.0)
    parser.add_argument("--min-dist", type=float, default=0.8, help="Minimum allowed MIC distance (angstrom).")
    parser.add_argument("--neighbor-update-steps", type=int, default=1, help="Update kNN every N steps.")
    parser.add_argument("--reduce-lattice", action="store_true", help="Apply simple lattice reduction.")
    parser.add_argument("--niggli-reduce", action="store_true", help="Apply Niggli reduction to lattices.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/samples_tokens"))
    parser.add_argument(
        "--z-sampling",
        type=str,
        default="temperature",
        choices=["argmax", "temperature", "topk", "topp"],
        help="Sampling strategy for element tokens.",
    )
    parser.add_argument("--z-temperature", type=float, default=1.2, help="Softmax temperature for z sampling.")
    parser.add_argument("--z-top-k", type=int, default=10, help="Top-k cutoff for z sampling.")
    parser.add_argument("--z-top-p", type=float, default=0.9, help="Top-p (nucleus) cutoff for z sampling.")
    parser.add_argument("--cell-init", type=str, default=None, choices=["gaussian", "iso"])
    parser.add_argument("--cell-init-scale", type=float, default=None)
    parser.add_argument("--cell-init-noise", type=float, default=None)
    return parser.parse_args()


def _load_npz_stats(npz_path: Path) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    data = np.load(npz_path)
    mask = data["atom_mask"]
    counts = mask.sum(axis=1).astype(int)
    counts = counts[counts > 0]
    lattice = data["lattice"] if "lattice" in data else None
    if lattice is None:
        return counts, None
    vols = np.abs(np.linalg.det(lattice))
    v_min = float(np.percentile(vols, 1.0))
    v_max = float(np.percentile(vols, 99.0))
    return counts, (v_min, v_max)


def _load_npz_scube_stats(npz_path: Path) -> Optional[Tuple[float, float, float, float]]:
    data = np.load(npz_path)
    lattice = data["lattice"] if "lattice" in data else None
    if lattice is None:
        return None
    det_l = np.abs(np.linalg.det(lattice))
    scube = np.power(det_l, 1.0 / 3.0)
    scube = scube[np.isfinite(scube)]
    if scube.size == 0:
        return None
    s10 = float(np.percentile(scube, 10.0))
    s50 = float(np.percentile(scube, 50.0))
    s90 = float(np.percentile(scube, 90.0))
    log_std = float(np.std(np.log(scube + 1e-12)))
    return s10, s50, s90, log_std


def _reduce_lattice_and_frac(lattice: np.ndarray, frac: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply simple lattice reduction (row reordering + handedness fix) and update
    fractional coordinates to preserve Cartesian positions.
    """
    lengths = np.linalg.norm(lattice, axis=1)
    order = np.argsort(lengths)
    reduced = lattice[order]
    if np.linalg.det(reduced) < 0:
        reduced[0] *= -1.0
    try:
        inv_reduced = np.linalg.inv(reduced)
        frac_new = frac @ lattice @ inv_reduced
    except np.linalg.LinAlgError:
        return lattice, frac
    frac_new = frac_new - np.floor(frac_new)
    return reduced, frac_new


def main() -> None:
    args = parse_args()
    if args.num_atoms is not None and args.num_atoms > args.max_atoms:
        raise ValueError("--num-atoms must be <= --max-atoms")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("config")
    if model_cfg is None:
        model_cfg = AtomTransformerConfig(num_elements=118, k_neighbors=32, g_scale=args.g_scale)
    else:
        if not hasattr(model_cfg, "cell_rep"):
            model_cfg.cell_rep = "gram6"
        if not hasattr(model_cfg, "chol_log_min"):
            model_cfg.chol_log_min = None
        if not hasattr(model_cfg, "chol_log_max"):
            model_cfg.chol_log_max = None
    diff_cfg = ckpt.get("diffusion_config")
    denoiser_cfg = AtomDenoiserConfig(model=model_cfg)
    if diff_cfg is not None:
        if isinstance(diff_cfg, dict):
            denoiser_cfg.diffusion = AtomDiffusionConfig(**diff_cfg)
        else:
            denoiser_cfg.diffusion = diff_cfg
        if not hasattr(denoiser_cfg.diffusion, "cell_rep"):
            denoiser_cfg.diffusion.cell_rep = "gram6"
        if not hasattr(denoiser_cfg.diffusion, "chol_log_min"):
            denoiser_cfg.diffusion.chol_log_min = None
        if not hasattr(denoiser_cfg.diffusion, "chol_log_max"):
            denoiser_cfg.diffusion.chol_log_max = None
        if not hasattr(denoiser_cfg.diffusion, "cell_init"):
            denoiser_cfg.diffusion.cell_init = "gaussian"
        if not hasattr(denoiser_cfg.diffusion, "cell_init_scale"):
            denoiser_cfg.diffusion.cell_init_scale = None
        if not hasattr(denoiser_cfg.diffusion, "cell_init_noise"):
            denoiser_cfg.diffusion.cell_init_noise = None
    if denoiser_cfg.diffusion.cell_rep == "cholesky6" and args.npz is not None:
        scube_stats = _load_npz_scube_stats(args.npz)
        if scube_stats is not None:
            s10, s50, s90, log_std = scube_stats
            if denoiser_cfg.diffusion.cell_init == "iso" and denoiser_cfg.diffusion.cell_init_scale is None:
                denoiser_cfg.diffusion.cell_init_scale = 1.5 * s50
            if denoiser_cfg.diffusion.cell_init_noise is None:
                denoiser_cfg.diffusion.cell_init_noise = float(min(max(log_std, 0.1), 0.2))
            if model_cfg.chol_log_min is None:
                model_cfg.chol_log_min = float(np.log(max(0.7 * s10, 1e-6)))
            if model_cfg.chol_log_max is None:
                model_cfg.chol_log_max = float(np.log(max(2.5 * s90, 1e-6)))
    if args.cell_init is not None:
        denoiser_cfg.diffusion.cell_init = args.cell_init
    if args.cell_init_scale is not None:
        denoiser_cfg.diffusion.cell_init_scale = args.cell_init_scale
    if args.cell_init_noise is not None:
        denoiser_cfg.diffusion.cell_init_noise = args.cell_init_noise
    denoiser_cfg.neighbor_update_steps = max(args.neighbor_update_steps, 1)
    # Avoid lattice-only reductions; apply coordinate-consistent reductions before export.
    denoiser_cfg.reduce_lattice = False
    denoiser_cfg.niggli_reduce = False
    model = AtomDenoiser(denoiser_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_counts = None
    vol_bounds = None
    if args.npz is not None:
        n_counts, vol_bounds = _load_npz_stats(args.npz)
    if args.num_atoms is None:
        if n_counts is None or len(n_counts) == 0:
            raise ValueError("num-atoms not set and no valid --npz stats found.")
        num_atoms_list = np.random.choice(n_counts, size=args.num_samples).astype(int).tolist()
    else:
        num_atoms_list = [args.num_atoms] * args.num_samples

    z_np = np.zeros((args.num_samples, args.max_atoms), dtype=np.int64)
    frac_np = np.zeros((args.num_samples, args.max_atoms, 3), dtype=np.float32)
    lattice_np = np.zeros((args.num_samples, 3, 3), dtype=np.float32)
    mask_np = np.zeros((args.num_samples, args.max_atoms), dtype=np.float32)

    for num_atoms in sorted(set(num_atoms_list)):
        idxs = [i for i, val in enumerate(num_atoms_list) if val == num_atoms]
        if not idxs:
            continue
        with torch.no_grad():
            z, frac, gram6, atom_mask = model.generate(
                num_atoms=num_atoms,
                max_atoms=args.max_atoms,
                batch_size=len(idxs),
                steps=args.steps,
                method=args.method,
                z_sampling=args.z_sampling,
                z_temperature=args.z_temperature,
                z_top_k=args.z_top_k,
                z_top_p=args.z_top_p,
            )
            lattice = model.gram6_to_lattice(gram6)
        z_np[idxs] = z.cpu().numpy()
        frac_np[idxs] = frac.cpu().numpy()
        lattice_np[idxs] = lattice.cpu().numpy()
        mask_np[idxs] = atom_mask.cpu().numpy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "samples.npz",
        z=z_np,
        frac=frac_np,
        lattice=lattice_np,
        atom_mask=mask_np,
    )

    valid = []
    element_counts = {}
    if vol_bounds is not None:
        v_min, v_max = vol_bounds
    else:
        v_min, v_max = None, None

    for i in range(args.num_samples):
        mask = (mask_np[i] > 0.5) & (z_np[i] > 0)
        zs = z_np[i][mask].astype(int).tolist()
        coords_np = frac_np[i][mask]
        lattice_mat = lattice_np[i]
        if not zs:
            continue

        if args.niggli_reduce:
            structure = Structure(
                lattice=lattice_mat,
                species=[Element.from_Z(z) for z in zs],
                coords=coords_np,
                coords_are_cartesian=False,
            )
            structure = structure.get_reduced_structure("niggli")
            lattice_mat = structure.lattice.matrix
            coords_np = np.asarray(structure.frac_coords, dtype=np.float32)
        elif args.reduce_lattice:
            lattice_mat, coords_np = _reduce_lattice_and_frac(lattice_mat, coords_np)

        lattice_np[i] = lattice_mat
        frac_np[i][mask] = coords_np

        for z_val in zs:
            element_counts[z_val] = element_counts.get(z_val, 0) + 1
        if v_min is not None and v_max is not None:
            vol = abs(np.linalg.det(lattice_mat))
            if vol < v_min or vol > v_max:
                valid.append(False)
                continue
        if len(zs) > 1:
            frac_t = torch.tensor(coords_np, device=device).unsqueeze(0)
            lat_t = torch.tensor(lattice_mat, device=device).unsqueeze(0)
            mask_t = torch.ones(1, frac_t.shape[1], device=device)
            dist = frac_mic_dist(frac_t, lat_t, mask_t)
            min_dist = torch.min(dist[0]).item()
            if min_dist < args.min_dist:
                valid.append(False)
                continue
        elements: List[Element] = [Element.from_Z(z) for z in zs]
        structure = Structure(lattice=lattice_mat, species=elements, coords=coords_np, coords_are_cartesian=False)
        CifWriter(structure).write_file(args.out_dir / f"sample_{i}.cif")
        valid.append(True)

    if valid:
        valid_rate = sum(valid) / len(valid)
        print(f"Saved {args.num_samples} samples to {args.out_dir} (valid_rate={valid_rate:.2f})")
        if element_counts:
            top = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top elements: {top}")
    else:
        print(f"Saved {args.num_samples} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
