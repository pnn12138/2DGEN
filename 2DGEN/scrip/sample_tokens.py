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
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights from checkpoint when available.")
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
    parser.add_argument(
        "--pbc-mask",
        type=str,
        default=None,
        help="Comma-separated PBC mask for MIC distance, e.g. 1,1,0 for slab.",
    )
    parser.add_argument(
        "--project-each-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Project frac/lattice back to valid manifold at every sampling step.",
    )
    parser.add_argument("--cond-npz", type=Path, default=None, help="NPZ with counts/lattice params for conditioning.")
    parser.add_argument("--cond-index", type=int, default=None, help="Use a specific row from cond-npz for all samples.")
    parser.add_argument("--cond-first", type=int, default=None, help="Use the first N rows from cond-npz.")
    parser.add_argument(
        "--cond-random",
        action="store_true",
        help="Sample random condition rows from cond-npz for each sample.",
    )
    parser.add_argument(
        "--save-cif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write sampled structures as CIF alongside samples.npz.",
    )
    parser.add_argument(
        "--cif-filter",
        type=str,
        default="all",
        choices=["all", "valid"],
        help="Which samples to export as CIF (all samples or only those passing validity checks).",
    )
    parser.add_argument(
        "--cif-mode",
        type=str,
        default="both",
        choices=["per-sample", "single", "both"],
        help="CIF output mode: per-sample files, a single multi-block CIF, or both.",
    )
    parser.add_argument(
        "--cif-prefix",
        type=str,
        default="sample",
        help="Prefix for per-sample CIF filenames (e.g. sample_0.cif).",
    )
    parser.add_argument(
        "--cif-filename",
        type=str,
        default="samples.cif",
        help="Filename for the single multi-block CIF (used when --cif-mode is single/both).",
    )
    return parser.parse_args()


def _parse_pbc_mask(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must have three comma-separated values, e.g. 1,1,0")
    mask = tuple(int(p) for p in parts)
    if any(p not in (0, 1) for p in mask):
        raise ValueError("--pbc-mask values must be 0 or 1")
    return mask  # type: ignore[return-value]


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
    g_scale = float(data["g_scale"]) if "g_scale" in data else 1.0
    det_l = np.abs(np.linalg.det(lattice))
    # For cholesky6, bounds apply to the *scaled* internal lattice (physical / sqrt(g_scale)).
    scube = np.power(det_l, 1.0 / 3.0) / max(g_scale, 1e-12) ** 0.5
    scube = scube[np.isfinite(scube)]
    if scube.size == 0:
        return None
    s10 = float(np.percentile(scube, 10.0))
    s50 = float(np.percentile(scube, 50.0))
    s90 = float(np.percentile(scube, 90.0))
    log_std = float(np.std(np.log(scube + 1e-12)))
    return s10, s50, s90, log_std


def _build_cond_from_npz(
    npz_path: Path,
    indices: np.ndarray,
    cond_fields: list[str],
    max_atoms: int,
    num_elements: int,
    cond_stats: dict | None = None,
) -> torch.Tensor:
    data = np.load(npz_path)
    parts = []
    for field in cond_fields:
        if field in ("counts", "counts_vector"):
            if "counts_vector" not in data:
                raise ValueError("counts_vector not found in cond npz.")
            counts = data["counts_vector"][indices].astype(np.float32)
            counts = counts / float(max_atoms)
            parts.append(counts)
            continue
        if field not in data:
            raise ValueError(f"{field} not found in cond npz.")
        value = data[field][indices].astype(np.float32)
        if value.ndim == 1:
            value = value[:, None]
        if cond_stats is not None and f"{field}_mean" in cond_stats and f"{field}_std" in cond_stats:
            mean = np.asarray(cond_stats[f"{field}_mean"], dtype=np.float32)
            std = np.asarray(cond_stats[f"{field}_std"], dtype=np.float32)
            value = (value - mean) / std
        parts.append(value)
    cond = np.concatenate(parts, axis=-1)
    if cond.shape[-1] == 0:
        raise ValueError("Condition vector has unexpected dimension.")
    return torch.from_numpy(cond)


def _cond_stats_from_npz(npz_path: Path, normalize_fields: list[str]) -> dict:
    data = np.load(npz_path)
    stats: dict[str, float | list] = {}
    for field in normalize_fields:
        if field in ("counts", "counts_vector"):
            continue
        mean_key = f"cond_{field}_mean"
        std_key = f"cond_{field}_std"
        if mean_key in data and std_key in data:
            stats[f"{field}_mean"] = data[mean_key].tolist()
            stats[f"{field}_std"] = data[std_key].tolist()
            continue
        if field in data:
            values = data[field].astype(np.float32)
            mean = values.mean(axis=0)
            std = values.std(axis=0)
            std = np.maximum(std, 1e-6)
            stats[f"{field}_mean"] = mean.tolist()
            stats[f"{field}_std"] = std.tolist()
    return stats


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
    n_counts = None
    vol_bounds = None
    if args.npz is not None:
        n_counts, vol_bounds = _load_npz_stats(args.npz)

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
        if not hasattr(model_cfg, "cond_dim"):
            model_cfg.cond_dim = 0
        if not hasattr(model_cfg, "pbc_mask"):
            model_cfg.pbc_mask = (1, 1, 0)
    if args.pbc_mask is not None:
        model_cfg.pbc_mask = _parse_pbc_mask(args.pbc_mask)
    diff_cfg = ckpt.get("diffusion_config")
    cond_cfg = ckpt.get("cond_config", {})
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
    if vol_bounds is not None:
        denoiser_cfg.v_min, denoiser_cfg.v_max = vol_bounds
    if args.cell_init is not None:
        denoiser_cfg.diffusion.cell_init = args.cell_init
    if args.cell_init_scale is not None:
        denoiser_cfg.diffusion.cell_init_scale = args.cell_init_scale
    if args.cell_init_noise is not None:
        denoiser_cfg.diffusion.cell_init_noise = args.cell_init_noise
    denoiser_cfg.neighbor_update_steps = max(args.neighbor_update_steps, 1)
    denoiser_cfg.project_each_step = args.project_each_step
    # Avoid lattice-only reductions; apply coordinate-consistent reductions before export.
    denoiser_cfg.reduce_lattice = False
    denoiser_cfg.niggli_reduce = False
    model = AtomDenoiser(denoiser_cfg).to(device)
    if args.use_ema and ckpt.get("ema_state_dict") is not None:
        model.load_state_dict(ckpt["ema_state_dict"], strict=False)
        print("Loaded EMA weights from checkpoint.")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cond = None
    cond_fields = []
    normalize_fields = []
    if isinstance(cond_cfg, dict):
        cond_fields = cond_cfg.get("cond_fields") or []
        normalize_fields = cond_cfg.get("cond_normalize_fields") or []
        if not cond_fields:
            if cond_cfg.get("include_lattice", False):
                cond_fields = ["counts_vector", "lattice_param"]
            else:
                cond_fields = ["counts_vector"]
            if cond_cfg.get("include_t", False):
                cond_fields.append("t")
    cond_stats = cond_cfg.get("cond_stats") if isinstance(cond_cfg, dict) else None
    if cond_cfg.get("use_condition"):
        cond_npz = args.cond_npz or args.npz
        if cond_npz is None:
            raise ValueError("Checkpoint expects conditioning; provide --cond-npz or --npz.")
        if not cond_stats:
            cond_stats = _cond_stats_from_npz(Path(cond_npz), normalize_fields)
        data = np.load(cond_npz)
        num_rows = data["counts_vector"].shape[0] if "counts_vector" in data else 0
        if num_rows == 0:
            raise ValueError("cond npz has no counts_vector rows.")
        if args.cond_first is not None:
            if args.cond_first <= 0:
                raise ValueError("--cond-first must be positive.")
            if args.cond_first > num_rows:
                raise ValueError("--cond-first exceeds rows available in cond npz.")
            if args.num_samples != args.cond_first:
                raise ValueError("--num-samples must equal --cond-first when using --cond-first.")
            indices = np.arange(args.cond_first, dtype=int)
        elif args.cond_index is not None:
            indices = np.full((args.num_samples,), args.cond_index, dtype=int)
        elif args.cond_random:
            rng = np.random.default_rng()
            indices = rng.integers(0, num_rows, size=args.num_samples, dtype=int)
        else:
            indices = np.zeros((args.num_samples,), dtype=int)
        cond = _build_cond_from_npz(
            Path(cond_npz),
            indices,
            cond_fields=cond_fields,
            max_atoms=cond_cfg.get("max_atoms", args.max_atoms),
            num_elements=cond_cfg.get("num_elements", 118),
            cond_stats=cond_stats,
        ).to(device)
        if model_cfg.cond_dim != cond.shape[-1]:
            raise ValueError(f"Condition dim {cond.shape[-1]} does not match model cond_dim {model_cfg.cond_dim}.")

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
                cond=cond[idxs] if cond is not None else None,
            )
            lattice = model.gram6_to_lattice(gram6)
        z_np[idxs] = z.cpu().numpy()
        frac_np[idxs] = frac.cpu().numpy()
        lattice_np[idxs] = lattice.cpu().numpy()
        mask_np[idxs] = atom_mask.cpu().numpy()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    valid_flags = np.zeros((args.num_samples,), dtype=np.int8)
    cif_written = np.zeros((args.num_samples,), dtype=np.int8)

    cif_blocks: list[str] = []
    element_counts = {}
    if vol_bounds is not None:
        v_min, v_max = vol_bounds
    else:
        v_min, v_max = None, None

    for i in range(args.num_samples):
        is_valid = True
        mask = (mask_np[i] > 0.5) & (z_np[i] > 0)
        zs = z_np[i][mask].astype(int).tolist()
        coords_np = frac_np[i][mask]
        lattice_mat = lattice_np[i]
        if not zs:
            is_valid = False
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

        coords_np = coords_np - np.floor(coords_np)
        lattice_np[i] = lattice_mat
        frac_np[i][mask] = coords_np

        if v_min is not None and v_max is not None:
            vol = abs(np.linalg.det(lattice_mat))
            if vol < v_min or vol > v_max:
                is_valid = False
        if len(zs) > 1:
            frac_t = torch.tensor(coords_np, device=device).unsqueeze(0)
            lat_t = torch.tensor(lattice_mat, device=device).unsqueeze(0)
            mask_t = torch.ones(1, frac_t.shape[1], device=device)
            dist = frac_mic_dist(frac_t, lat_t, mask_t, pbc_mask=model_cfg.pbc_mask)
            min_dist = torch.min(dist[0]).item()
            if min_dist < args.min_dist:
                is_valid = False
        elements: List[Element] = [Element.from_Z(z) for z in zs]
        structure = Structure(lattice=lattice_mat, species=elements, coords=coords_np, coords_are_cartesian=False)

        if is_valid:
            valid_flags[i] = 1
            for z_val in zs:
                element_counts[z_val] = element_counts.get(z_val, 0) + 1

        if args.save_cif:
            if args.cif_filter == "valid" and not is_valid:
                continue
            try:
                writer = CifWriter(structure)
                if args.cif_mode in ("per-sample", "both"):
                    writer.write_file(args.out_dir / f"{args.cif_prefix}_{i}.cif")
                if args.cif_mode in ("single", "both"):
                    cif_str = str(writer).rstrip()
                    lines = cif_str.splitlines()
                    for line_idx, line in enumerate(lines):
                        if line.startswith("data_"):
                            lines[line_idx] = f"data_{args.cif_prefix}_{i:04d}"
                            break
                    cif_blocks.append("\n".join(lines).rstrip())
                cif_written[i] = 1
            except Exception:
                cif_written[i] = 0

    if args.save_cif and args.cif_mode in ("single", "both") and args.cif_filename:
        if cif_blocks:
            (args.out_dir / args.cif_filename).write_text("\n\n".join(cif_blocks).rstrip() + "\n")

    np.savez_compressed(
        args.out_dir / "samples.npz",
        z=z_np,
        frac=frac_np,
        lattice=lattice_np,
        atom_mask=mask_np,
        valid=valid_flags,
        cif_written=cif_written,
    )

    valid_rate = float(np.mean(valid_flags)) if valid_flags.size else 0.0
    cif_rate = float(np.mean(cif_written)) if cif_written.size else 0.0
    print(
        f"Saved {args.num_samples} samples to {args.out_dir} "
        f"(valid_rate={valid_rate:.2f}, cif_rate={cif_rate:.2f})"
    )
    if element_counts:
        top = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top elements: {top}")


if __name__ == "__main__":
    main()
