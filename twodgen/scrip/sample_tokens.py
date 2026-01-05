from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter

from twodgen.common.crystal import frac_mic_dist
from twodgen.evaluate import eval_samples as eval_samples_mod
from twodgen.model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig
from twodgen.common.atom_diffusion import AtomDiffusionConfig
from twodgen.model.atom_transformer import AtomTransformerConfig


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample token-based crystal diffusion and export CIF.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic algorithms (may be slower).",
    )
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
        "--coord-frame",
        type=str,
        default="canon",
        choices=["raw", "canon"],
        help="Coordinate frame of frac/lattice for sampling and outputs.",
    )
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
    parser.add_argument(
        "--project-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Update uv_angle/z_norm/lattice_param/t; project uv_angle/z_norm back to valid manifold each step.",
    )
    parser.add_argument(
        "--z-norm-clip",
        type=float,
        default=None,
        help="Clip range for z_norm projection (defaults to npz value or 1.5).",
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
    parser.add_argument("--eval", action="store_true", help="Run eval after sampling and write metrics.")
    parser.add_argument("--eval-out-dir", type=Path, default=None, help="Output directory for eval artifacts.")
    parser.add_argument("--eval-stats-npz", type=Path, default=None, help="NPZ for volume bounds (p1/p99).")
    parser.add_argument("--eval-min-dist", type=float, default=1.5)
    parser.add_argument("--eval-bond-cut", type=float, default=3.0)
    parser.add_argument("--eval-dup-eps", type=float, default=1e-3)
    parser.add_argument("--eval-v-min", type=float, default=None)
    parser.add_argument("--eval-v-max", type=float, default=None)
    parser.add_argument(
        "--eval-pbc-mask",
        type=str,
        default=None,
        help="Override PBC mask for evaluation (default uses model config).",
    )
    return parser.parse_args(argv)


def _parse_pbc_mask(value: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must have three comma-separated values, e.g. 1,1,0")
    mask = tuple(int(p) for p in parts)
    if any(p not in (0, 1) for p in mask):
        raise ValueError("--pbc-mask values must be 0 or 1")
    return mask  # type: ignore[return-value]


def _load_npz_stats(
    npz_path: Path, coord_frame: str
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    data = np.load(npz_path)
    mask = data["atom_mask"]
    counts = mask.sum(axis=1).astype(int)
    counts = counts[counts > 0]
    if coord_frame == "canon" and "lattice_canon" in data:
        lattice = data["lattice_canon"]
    else:
        lattice = data["lattice"] if "lattice" in data else None
    if lattice is None:
        return counts, None
    vols = np.abs(np.linalg.det(lattice))
    v_min = float(np.percentile(vols, 1.0))
    v_max = float(np.percentile(vols, 99.0))
    return counts, (v_min, v_max)


def _load_npz_scube_stats(npz_path: Path, coord_frame: str) -> Optional[Tuple[float, float, float, float]]:
    data = np.load(npz_path)
    if coord_frame == "canon" and "lattice_canon" in data:
        lattice = data["lattice_canon"]
    else:
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


def _load_npz_z_norm_clip(npz_path: Path) -> Optional[float]:
    data = np.load(npz_path)
    if "z_norm_clip" not in data:
        return None
    return float(np.asarray(data["z_norm_clip"]).reshape(-1)[0])


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


def run_sampling(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    if args.num_atoms is not None and args.num_atoms > args.max_atoms:
        raise ValueError("--num-atoms must be <= --max-atoms")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_counts = None
    vol_bounds = None
    if args.npz is not None:
        n_counts, vol_bounds = _load_npz_stats(args.npz, coord_frame=args.coord_frame)

    print("Warning: loading checkpoint with torch.load (weights_only=False). Use only trusted checkpoints.")
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
        if not hasattr(model_cfg, "use_comp_encoder"):
            model_cfg.use_comp_encoder = True
        if not hasattr(model_cfg, "comp_embed_dim"):
            model_cfg.comp_embed_dim = 64
        if not hasattr(model_cfg, "comp_pool_mode"):
            model_cfg.comp_pool_mode = "count"
        if not hasattr(model_cfg, "comp_use_frac"):
            model_cfg.comp_use_frac = True
        if not hasattr(model_cfg, "element_ids"):
            model_cfg.element_ids = None
        if not hasattr(model_cfg, "pbc_mask"):
            model_cfg.pbc_mask = (1, 1, 0)
        if not hasattr(model_cfg, "dual_graph"):
            model_cfg.dual_graph = False
        if not hasattr(model_cfg, "edge_type_dim"):
            model_cfg.edge_type_dim = 0
        if not hasattr(model_cfg, "edge_type_gating"):
            model_cfg.edge_type_gating = True
        if not hasattr(model_cfg, "wrap_embed_dim"):
            model_cfg.wrap_embed_dim = 0
    if args.pbc_mask is not None:
        model_cfg.pbc_mask = _parse_pbc_mask(args.pbc_mask)
    diff_cfg = ckpt.get("diffusion_config")
    cond_cfg = ckpt.get("cond_config", {})
    geom_cfg = ckpt.get("geometry_config", {})
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
        scube_stats = _load_npz_scube_stats(args.npz, coord_frame=args.coord_frame)
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
    denoiser_cfg.project_geometry = args.project_geometry
    z_norm_clip = args.z_norm_clip
    if z_norm_clip is None and args.npz is not None:
        z_norm_clip = _load_npz_z_norm_clip(args.npz)
    if z_norm_clip is None and args.cond_npz is not None:
        z_norm_clip = _load_npz_z_norm_clip(args.cond_npz)
    if z_norm_clip is None:
        z_norm_clip = denoiser_cfg.z_norm_clip
    denoiser_cfg.z_norm_clip = float(z_norm_clip)
    if args.project_geometry and isinstance(geom_cfg, dict):
        if geom_cfg.get("use_geometry_fields") is False:
            print("[warn] geometry projection requested but checkpoint was trained without geometry heads.")
    # Avoid lattice-only reductions; apply coordinate-consistent reductions before export.
    denoiser_cfg.reduce_lattice = False
    denoiser_cfg.niggli_reduce = False
    model = AtomDenoiser(denoiser_cfg).to(device)
    if args.use_ema and ckpt.get("ema_state_dict") is not None:
        incompatible = model.load_state_dict(ckpt["ema_state_dict"], strict=False)
        print("Loaded EMA weights from checkpoint.")
    else:
        incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"[warn] Missing keys in checkpoint: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"[warn] Unexpected keys in checkpoint: {len(incompatible.unexpected_keys)}")
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
    t_normalize = False
    t_mean = None
    t_std = None
    if isinstance(geom_cfg, dict):
        t_normalize = bool(geom_cfg.get("t_normalize", False))
        t_mean = geom_cfg.get("t_mean")
        t_std = geom_cfg.get("t_std")
    rng = np.random.default_rng(args.seed)

    cond_counts_vector = None
    cond_indices = None
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
            indices = rng.integers(0, num_rows, size=args.num_samples, dtype=int)
        else:
            indices = np.zeros((args.num_samples,), dtype=int)
        cond_indices = indices.astype(np.int64)
        if "counts_vector" in data and ("counts_vector" in cond_fields or "counts" in cond_fields):
            cond_counts_vector = data["counts_vector"][indices].astype(np.int64)
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
        if cond_counts_vector is not None:
            num_atoms_list = cond_counts_vector.sum(axis=1).astype(int).tolist()
            if len(num_atoms_list) != args.num_samples:
                raise ValueError("cond counts size does not match --num-samples.")
        else:
            if n_counts is None or len(n_counts) == 0:
                raise ValueError("num-atoms not set and no valid --npz stats found.")
            num_atoms_list = rng.choice(n_counts, size=args.num_samples).astype(int).tolist()
    else:
        num_atoms_list = [args.num_atoms] * args.num_samples
    if any(n > args.max_atoms for n in num_atoms_list):
        raise ValueError("Sampled num-atoms exceeds --max-atoms.")

    z_np = np.zeros((args.num_samples, args.max_atoms), dtype=np.int64)
    frac_np = np.zeros((args.num_samples, args.max_atoms, 3), dtype=np.float32)
    lattice_np = np.zeros((args.num_samples, 3, 3), dtype=np.float32)
    mask_np = np.zeros((args.num_samples, args.max_atoms), dtype=np.float32)
    lattice_param_np = None
    slab_t_np = None

    for num_atoms in sorted(set(num_atoms_list)):
        idxs = [i for i, val in enumerate(num_atoms_list) if val == num_atoms]
        if not idxs:
            continue
        with torch.no_grad():
            counts_tensor = None
            if cond_counts_vector is not None:
                counts_tensor = torch.from_numpy(cond_counts_vector[idxs]).to(device)
            z, frac, gram6, atom_mask, lattice_param, slab_t = model.generate(
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
                counts_vector=counts_tensor,
            )
            lattice = model.gram6_to_lattice(gram6)
        z_np[idxs] = z.cpu().numpy()
        frac_np[idxs] = frac.cpu().numpy()
        lattice_np[idxs] = lattice.cpu().numpy()
        mask_np[idxs] = atom_mask.cpu().numpy()
        if lattice_param is not None:
            if lattice_param_np is None:
                lattice_param_np = np.zeros((args.num_samples, lattice_param.shape[-1]), dtype=np.float32)
            lattice_param_np[idxs] = lattice_param.cpu().numpy()
        if slab_t is not None:
            if slab_t_np is None:
                slab_t_np = np.zeros((args.num_samples,), dtype=np.float32)
            slab_t_out = slab_t
            if t_normalize and t_mean is not None and t_std is not None:
                slab_t_out = slab_t_out * float(t_std) + float(t_mean)
            slab_t_np[idxs] = slab_t_out.cpu().numpy()

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

    payload = {
        "z": z_np,
        "frac": frac_np,
        "lattice": lattice_np,
        "atom_mask": mask_np,
        "valid": valid_flags,
        "cif_written": cif_written,
        "coord_frame": np.array(args.coord_frame),
    }
    if lattice_param_np is not None:
        payload["lattice_param"] = lattice_param_np
    if slab_t_np is not None:
        payload["t"] = slab_t_np
    if cond_indices is not None:
        payload["cond_indices"] = cond_indices
    if cond_counts_vector is not None:
        payload["cond_counts_vector"] = cond_counts_vector
    np.savez_compressed(args.out_dir / "samples.npz", **payload)

    valid_rate = float(np.mean(valid_flags)) if valid_flags.size else 0.0
    cif_rate = float(np.mean(cif_written)) if cif_written.size else 0.0
    print(
        f"Saved {args.num_samples} samples to {args.out_dir} "
        f"(valid_rate={valid_rate:.2f}, cif_rate={cif_rate:.2f})"
    )
    if element_counts:
        top = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top elements: {top}")
    if args.eval:
        samples_path = args.out_dir / "samples.npz"
        samples = np.load(samples_path)
        eval_out_dir = args.eval_out_dir or (args.out_dir / "eval")
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        eval_pbc_mask = model_cfg.pbc_mask
        if args.eval_pbc_mask is not None:
            eval_pbc_mask = _parse_pbc_mask(args.eval_pbc_mask)
        v_min = args.eval_v_min
        v_max = args.eval_v_max
        if args.eval_stats_npz is not None:
            stats = eval_samples_mod._load_npz_stats(args.eval_stats_npz)
            if stats is not None:
                v_min, v_max = stats
        per_sample, tier0, tier1 = eval_samples_mod._eval_samples(
            samples,
            v_min=v_min,
            v_max=v_max,
            min_dist_cut=args.eval_min_dist,
            bond_cut=args.eval_bond_cut,
            dup_eps=args.eval_dup_eps,
            pbc_mask=eval_pbc_mask,
        )
        with (eval_out_dir / "per_sample.jsonl").open("w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        with (eval_out_dir / "tier0_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(tier0, f, indent=2, ensure_ascii=True)
        with (eval_out_dir / "tier1_2d_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(tier1, f, indent=2, ensure_ascii=True)
        print(f"Saved eval outputs to {eval_out_dir}")
    return args.out_dir / "samples.npz"


def main() -> None:
    args = parse_args()
    run_sampling(args)


if __name__ == "__main__":
    main()
