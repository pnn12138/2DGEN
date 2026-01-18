from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple
import sys

import numpy as np
import torch
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter

from twodgen.common.crystal import frac_mic_dist
from twodgen.evaluate import eval_samples as eval_samples_mod
from twodgen.common.run_metadata import collect_run_metadata
from twodgen.data.splits import load_c2db_split, select_split_indices, validate_split_indices
from twodgen.model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig
from twodgen.common.atom_diffusion import AtomDiffusionConfig
from twodgen.model.atom_transformer import AtomTransformerConfig


def _install_checkpoint_legacy_shims() -> None:
    """
    Install legacy module name aliases to load older checkpoints.

    Historical checkpoints stored config objects under module paths like
    `model.atom_transformer.AtomTransformerConfig`. The codebase has since moved
    them to `twodgen.*`. We alias those modules so `torch.load(..., weights_only=False)`
    can unpickle trusted checkpoints without requiring the old package layout.
    """
    import sys
    import types

    import twodgen.common.atom_diffusion as atom_diffusion_mod
    import twodgen.model.atom_denoiser as atom_denoiser_mod
    import twodgen.model.atom_transformer as atom_transformer_mod

    sys.modules.setdefault("model", types.ModuleType("model"))
    sys.modules.setdefault("common", types.ModuleType("common"))
    sys.modules["model.atom_transformer"] = atom_transformer_mod
    sys.modules["model.atom_denoiser"] = atom_denoiser_mod
    sys.modules["common.atom_diffusion"] = atom_diffusion_mod


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample token-based crystal diffusion and export CIF.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npz", type=Path, default=None, help="Token cache for sampling N/volume stats.")
    parser.add_argument("--cond-npz", type=Path, default=None, help="Token cache used to provide conditioning rows.")
    parser.add_argument(
        "--cond-split-json",
        type=Path,
        default=None,
        help="Optional split json to restrict conditioning rows (train/heldout indices).",
    )
    parser.add_argument(
        "--cond-split",
        type=str,
        default="all",
        choices=["all", "train", "heldout"],
        help="Which subset to draw conditioning rows from when --cond-split-json is set.",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/samples_tokens"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--project-each-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, project fractional coords and clip lattice every sampling step.",
    )
    parser.add_argument(
        "--vacuum-min",
        type=float,
        default=None,
        help="If set, mark samples invalid when vacuum thickness (Angstrom) is below this threshold.",
    )
    parser.add_argument(
        "--reject-cross-vacuum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, mark samples invalid when a bond crosses the vacuum axis under 3D PBC.",
    )

    parser.set_defaults(
        deterministic=False,
        use_ema=True,
        max_atoms=24,
        num_atoms=None,
        g_scale=100.0,
        min_dist=None,
        min_dist_project=True,
        min_dist_iter=8,
        min_dist_strength=0.03,
        min_dist_cut=None,
        neighbor_update_steps=1,
        reduce_lattice=False,
        niggli_reduce=False,
        z_sampling="temperature",
        z_temperature=1.2,
        z_top_k=10,
        z_top_p=0.9,
        cell_init=None,
        cell_init_scale=None,
        cell_init_noise=None,
        coord_frame="canon",
        pbc_mask=None,
        project_each_step=True,
        project_geometry=True,
        z_norm_clip=None,
        cond_index=None,
        cond_first=None,
        cond_random=True,
        save_cif=True,
        cif_filter="all",
        cif_mode="both",
        cif_prefix="sample",
        cif_filename="samples.cif",
        eval=True,
        eval_out_dir=None,
        eval_stats_npz=None,
        eval_min_dist=1.5,
        eval_bond_cut=3.0,
        eval_dup_eps=1e-3,
        eval_v_min=None,
        eval_v_max=None,
        eval_pbc_mask=None,
        vacuum_min=None,
        reject_cross_vacuum=False,
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

    sampling_config: dict = {
        "run_metadata": collect_run_metadata(argv=sys.argv),
        "checkpoint": str(args.checkpoint),
        "npz": str(args.npz) if args.npz is not None else None,
        "cond_npz": str(args.cond_npz) if args.cond_npz is not None else None,
        "cond_split_json": str(args.cond_split_json) if args.cond_split_json is not None else None,
        "cond_split": str(args.cond_split),
        "num_samples": int(args.num_samples),
        "steps": int(args.steps),
        "method": str(args.method),
        "seed": int(args.seed),
    }

    min_dist_cut = float(args.eval_min_dist)
    if args.min_dist is not None:
        if args.eval_min_dist != 1.5 and args.eval_min_dist != args.min_dist:
            print(
                "[warn] Both --min-dist and --eval-min-dist provided; using --min-dist for validity checks."
            )
        print("[warn] --min-dist is deprecated; use --eval-min-dist instead.")
        min_dist_cut = float(args.min_dist)
        args.eval_min_dist = float(args.min_dist)
    min_dist_project = bool(args.min_dist_project)
    min_dist_iter = max(int(args.min_dist_iter), 0)
    if min_dist_project and min_dist_iter == 0:
        min_dist_iter = 5
        print("[info] --min-dist-project enabled; defaulting --min-dist-iter to 5.")
    min_dist_strength = float(args.min_dist_strength)
    min_dist_repulsion_cut = float(args.min_dist_cut) if args.min_dist_cut is not None else min_dist_cut

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_counts = None
    vol_bounds = None
    coord_frame_actual = args.coord_frame
    if args.coord_frame == "canon" and args.npz is not None:
        data = np.load(args.npz)
        if "f_canon" not in data or "lattice_canon" not in data:
            coord_frame_actual = "raw"
            print(
                "[warn] coord_frame=canon requested but npz lacks f_canon/lattice_canon; "
                "falling back to raw coord frame."
            )
    args.coord_frame_actual = coord_frame_actual
    if args.npz is not None:
        n_counts, vol_bounds = _load_npz_stats(args.npz, coord_frame=coord_frame_actual)

    print("Warning: loading checkpoint with torch.load (weights_only=False). Use only trusted checkpoints.")
    _install_checkpoint_legacy_shims()
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
    print(
        "[info] checkpoint_config "
        f"cell_rep={model_cfg.cell_rep} g_scale={model_cfg.g_scale} pbc_mask={model_cfg.pbc_mask}"
    )
    if args.g_scale != model_cfg.g_scale:
        print(f"[warn] args.g_scale={args.g_scale} differs from checkpoint g_scale={model_cfg.g_scale}")
    diff_cfg = ckpt.get("diffusion_config")
    cond_cfg = ckpt.get("cond_config", {})
    geom_cfg = ckpt.get("geometry_config", {})
    denoiser_cfg = AtomDenoiserConfig(model=model_cfg)
    denoiser_cfg.min_dist_iter = min_dist_iter if min_dist_project else 0
    denoiser_cfg.min_dist_strength = min_dist_strength
    denoiser_cfg.min_dist_cut = min_dist_repulsion_cut
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
        scube_stats = _load_npz_scube_stats(args.npz, coord_frame=coord_frame_actual)
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
    geom_use_fields = None
    if isinstance(geom_cfg, dict):
        geom_use_fields = geom_cfg.get("use_geometry_fields")
    print(f"[info] geometry_use_fields={geom_use_fields}, project_geometry={args.project_geometry}")
    if args.project_geometry and isinstance(geom_cfg, dict):
        if geom_cfg.get("use_geometry_fields") is False:
            raise ValueError(
                "--project-geometry requested but checkpoint was trained without geometry heads. "
                "Disable --project-geometry or retrain with --use-geometry-fields."
            )
    if args.project_geometry and not isinstance(geom_cfg, dict):
        print("[warn] geometry config not found in checkpoint; --project-geometry may be unsafe.")
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
    cond_strategy = None
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
        pool_indices: Optional[list[int]] = None
        pool_set: Optional[set[int]] = None
        if args.cond_split_json is not None and args.cond_split != "all":
            split = load_c2db_split(args.cond_split_json)
            pool_indices = select_split_indices(split, args.cond_split)
            if not pool_indices:
                raise ValueError(f"cond_split {args.cond_split!r} is empty in {args.cond_split_json}.")
            validate_split_indices(pool_indices, total=num_rows)
            pool_set = set(pool_indices)
            sampling_config["cond_pool_size"] = int(len(pool_indices))
        else:
            sampling_config["cond_pool_size"] = int(num_rows)
        if args.cond_first is not None:
            if args.cond_first <= 0:
                raise ValueError("--cond-first must be positive.")
            if pool_indices is None and args.cond_first > num_rows:
                raise ValueError("--cond-first exceeds rows available in cond npz.")
            if args.num_samples != args.cond_first:
                raise ValueError("--num-samples must equal --cond-first when using --cond-first.")
            cond_strategy = "first"
            if pool_indices is None:
                indices = np.arange(args.cond_first, dtype=int)
            else:
                indices = np.asarray(pool_indices[: args.cond_first], dtype=int)
        elif args.cond_index is not None:
            if args.cond_index < 0 or args.cond_index >= num_rows:
                raise ValueError("--cond-index is out of range for cond npz.")
            if pool_set is not None and int(args.cond_index) not in pool_set:
                raise ValueError("--cond-index is not part of the requested --cond-split subset.")
            cond_strategy = "index"
            indices = np.full((args.num_samples,), args.cond_index, dtype=int)
        elif args.cond_random:
            cond_strategy = "random"
            if pool_indices is None:
                indices = rng.integers(0, num_rows, size=args.num_samples, dtype=int)
            else:
                pool = np.asarray(pool_indices, dtype=int)
                picked = rng.integers(0, len(pool), size=args.num_samples, dtype=int)
                indices = pool[picked]
        else:
            cond_strategy = "random"
            print("[info] No cond strategy provided; defaulting to --cond-random.")
            if pool_indices is None:
                indices = rng.integers(0, num_rows, size=args.num_samples, dtype=int)
            else:
                pool = np.asarray(pool_indices, dtype=int)
                picked = rng.integers(0, len(pool), size=args.num_samples, dtype=int)
                indices = pool[picked]
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
        sampling_config["cond_strategy"] = str(cond_strategy)

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
    if cond_strategy is not None:
        unique_atoms, counts = np.unique(np.asarray(num_atoms_list), return_counts=True)
        summary = {int(k): int(v) for k, v in zip(unique_atoms.tolist(), counts.tolist())}
        print(f"[info] cond_strategy={cond_strategy}, n_atoms_hist={summary}")

    z_np = np.zeros((args.num_samples, args.max_atoms), dtype=np.int64)
    frac_np = np.zeros((args.num_samples, args.max_atoms, 3), dtype=np.float32)
    lattice_np = np.zeros((args.num_samples, 3, 3), dtype=np.float32)
    mask_np = np.zeros((args.num_samples, args.max_atoms), dtype=np.float32)
    min_dist_pre_np = np.full((args.num_samples,), np.nan, dtype=np.float32)
    min_dist_post_np = np.full((args.num_samples,), np.nan, dtype=np.float32)
    thickness_np = np.full((args.num_samples,), np.nan, dtype=np.float32)
    vacuum_np = np.full((args.num_samples,), np.nan, dtype=np.float32)
    cross_vacuum_np = np.zeros((args.num_samples,), dtype=np.int8)
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
            (
                z,
                frac,
                gram6,
                atom_mask,
                lattice_param,
                slab_t,
                min_dist_pre,
                min_dist_post,
            ) = model.generate(
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
        min_dist_pre_np[idxs] = min_dist_pre.cpu().numpy()
        min_dist_post_np[idxs] = min_dist_post.cpu().numpy()
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

        lengths = np.linalg.norm(lattice_mat, axis=1)
        c_idx = int(np.argmax(lengths)) if np.all(np.isfinite(lengths)) else 2
        c_len = float(lengths[c_idx]) if np.all(np.isfinite(lengths)) else float("nan")
        thickness, vacuum = eval_samples_mod._thickness_vacuum(coords_np[:, c_idx], c_len)
        thickness_np[i] = float(thickness)
        vacuum_np[i] = float(vacuum)
        if args.reject_cross_vacuum and len(zs) > 1:
            dist_3d, _, shifts_3d = eval_samples_mod._min_dist_and_shifts(
                coords_np, lattice_mat, pbc_mask=(1, 1, 1)
            )
            if np.ndim(dist_3d) == 0:
                cross_vacuum_np[i] = 0
                continue
            edges = np.where(dist_3d < float(args.eval_bond_cut))
            cross_vac = False
            for a, b in zip(edges[0].tolist(), edges[1].tolist()):
                if a >= b:
                    continue
                if abs(float(shifts_3d[a, b, c_idx])) > 0.0:
                    cross_vac = True
                    break
            cross_vacuum_np[i] = int(cross_vac)
            if cross_vac:
                is_valid = False
        if args.vacuum_min is not None and np.isfinite(vacuum) and float(vacuum) < float(args.vacuum_min):
            is_valid = False

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
            if min_dist < min_dist_cut:
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

    valid_rate = float(np.mean(valid_flags)) if valid_flags.size else 0.0
    cif_rate = float(np.mean(cif_written)) if cif_written.size else 0.0
    pre_collision = int(np.sum(min_dist_pre_np < min_dist_cut))
    post_collision = int(np.sum(min_dist_post_np < min_dist_cut))
    pre_mean = float(np.nanmean(min_dist_pre_np)) if min_dist_pre_np.size else float("nan")
    post_mean = float(np.nanmean(min_dist_post_np)) if min_dist_post_np.size else float("nan")
    pre_p10 = float(np.nanpercentile(min_dist_pre_np, 10.0)) if min_dist_pre_np.size else float("nan")
    post_p10 = float(np.nanpercentile(min_dist_post_np, 10.0)) if min_dist_post_np.size else float("nan")
    payload = {
        "z": z_np,
        "frac": frac_np,
        "lattice": lattice_np,
        "atom_mask": mask_np,
        "valid": valid_flags,
        "cif_written": cif_written,
        "coord_frame": np.array(args.coord_frame),
        "coord_frame_actual": np.array(getattr(args, "coord_frame_actual", args.coord_frame)),
        "min_dist_cut": np.array(min_dist_cut, dtype=np.float32),
        "min_dist_repulsion_cut": np.array(min_dist_repulsion_cut, dtype=np.float32),
        "min_dist_repulsion_iter": np.array(min_dist_iter, dtype=np.int64),
        "min_dist_repulsion_strength": np.array(min_dist_strength, dtype=np.float32),
        "valid_rate": np.array(valid_rate, dtype=np.float32),
        "min_dist_pre": min_dist_pre_np,
        "min_dist_post": min_dist_post_np,
        "thickness": thickness_np,
        "vacuum": vacuum_np,
        "cross_vacuum_bond": cross_vacuum_np,
    }
    if lattice_param_np is not None:
        payload["lattice_param"] = lattice_param_np
    if slab_t_np is not None:
        payload["t"] = slab_t_np
    if cond_indices is not None:
        payload["cond_indices"] = cond_indices
    if cond_counts_vector is not None:
        payload["cond_counts_vector"] = cond_counts_vector
    if cond_strategy is not None:
        payload["cond_strategy"] = np.array(cond_strategy)
    if args.cond_split_json is not None:
        payload["cond_split"] = np.array(str(args.cond_split))
    np.savez_compressed(args.out_dir / "samples.npz", **payload)
    sampling_config["export"] = {
        "valid_rate_samples": float(valid_rate),
        "cif_rate_samples": float(cif_rate),
        "min_dist_pre_mean": float(pre_mean),
        "min_dist_post_mean": float(post_mean),
        "min_dist_pre_p10": float(pre_p10),
        "min_dist_post_p10": float(post_p10),
        "collision_pre": int(pre_collision),
        "collision_post": int(post_collision),
        "min_dist_cut": float(min_dist_cut),
        "min_dist_repulsion_cut": float(min_dist_repulsion_cut),
        "min_dist_repulsion_iter": int(min_dist_iter),
        "min_dist_repulsion_strength": float(min_dist_strength),
    }
    if cond_counts_vector is not None:
        num_elements = int(cond_counts_vector.shape[-1])
        gen_counts = np.zeros((args.num_samples, num_elements), dtype=np.int64)
        valid_atoms = (mask_np > 0.5) & (z_np > 0)
        for i in range(args.num_samples):
            zs = z_np[i][valid_atoms[i]].astype(int)
            if zs.size == 0:
                continue
            idx = zs - 1
            idx = idx[(idx >= 0) & (idx < num_elements)]
            if idx.size:
                np.add.at(gen_counts[i], idx, 1)
        target_counts = cond_counts_vector.astype(np.int64)
        l1 = np.abs(gen_counts - target_counts).sum(axis=-1)
        denom = np.linalg.norm(gen_counts, axis=-1) * np.linalg.norm(target_counts, axis=-1)
        denom = np.maximum(denom, 1e-12)
        cos = (gen_counts * target_counts).sum(axis=-1) / denom
        sampling_config["composition"] = {
            "hit_rate": float(np.mean(l1 == 0)),
            "l1_mean": float(np.mean(l1)),
            "l1_median": float(np.median(l1)),
            "cos_mean": float(np.mean(cos)),
        }
    (args.out_dir / "sampling_config.json").write_text(
        json.dumps(sampling_config, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(
        f"Saved {args.num_samples} samples to {args.out_dir} "
        f"(valid_rate={valid_rate:.2f}, cif_rate={cif_rate:.2f})"
    )
    print(
        "[info] min_dist pre/post repulsion: "
        f"mean={pre_mean:.3f}/{post_mean:.3f}, "
        f"p10={pre_p10:.3f}/{post_p10:.3f}, "
        f"collision={pre_collision}/{post_collision}"
    )
    print(f"[info] min_dist_cut={min_dist_cut:.3f}, eval_min_dist={args.eval_min_dist:.3f}")
    print(
        f"[info] min_dist_repulsion=({min_dist_project}), "
        f"iter={min_dist_iter}, strength={min_dist_strength:.3f}, cut={min_dist_repulsion_cut:.3f}"
    )
    if element_counts:
        top = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top elements: {top}")
    if args.eval:
        samples_path = args.out_dir / "samples.npz"
        samples = np.load(samples_path)
        eval_out_dir = args.eval_out_dir or (args.out_dir / "eval")
        eval_pbc_mask = model_cfg.pbc_mask
        if args.eval_pbc_mask is not None:
            eval_pbc_mask = _parse_pbc_mask(args.eval_pbc_mask)
        v_min = args.eval_v_min
        v_max = args.eval_v_max
        stats_npz = args.eval_stats_npz
        if stats_npz is None and args.npz is not None:
            stats_npz = args.npz
        if stats_npz is not None:
            stats = eval_samples_mod._load_npz_stats(stats_npz)
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
            vacuum_min=args.vacuum_min,
        )
        eval_params = eval_samples_mod.build_eval_params(
            min_dist_cut=float(args.eval_min_dist),
            bond_cut=float(args.eval_bond_cut),
            dup_eps=float(args.eval_dup_eps),
            vacuum_min=args.vacuum_min,
            v_min=v_min,
            v_max=v_max,
            pbc_mask=eval_pbc_mask,
        )
        eval_samples_mod.write_eval_outputs(
            out_dir=eval_out_dir,
            per_sample=per_sample,
            tier0=tier0,
            tier1=tier1,
            eval_params=eval_params,
            run_context={
                "source": "sample_tokens",
                "samples": str(samples_path),
                "cond_split": str(args.cond_split),
                "cond_split_json": str(args.cond_split_json) if args.cond_split_json is not None else None,
            },
        )
        print(f"Saved eval outputs to {eval_out_dir}")
    return args.out_dir / "samples.npz"


def main() -> None:
    args = parse_args()
    run_sampling(args)


if __name__ == "__main__":
    main()
