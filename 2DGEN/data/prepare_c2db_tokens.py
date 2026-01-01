from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from data.preprocess import PreprocessConfig, preprocess_cartesian  # noqa: E402


def _pad_1d(values: np.ndarray, max_len: int, pad_value: float) -> Tuple[np.ndarray, np.ndarray]:
    trimmed = values[:max_len]
    padded = np.full((max_len,), pad_value, dtype=trimmed.dtype)
    padded[: len(trimmed)] = trimmed
    mask = np.zeros((max_len,), dtype=np.float32)
    mask[: len(trimmed)] = 1.0
    return padded, mask


def _pad_2d(values: np.ndarray, max_len: int, pad_value: float) -> Tuple[np.ndarray, np.ndarray]:
    trimmed = values[:max_len]
    padded = np.full((max_len, 3), pad_value, dtype=np.float32)
    padded[: len(trimmed)] = trimmed.astype(np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    mask[: len(trimmed)] = 1.0
    return padded, mask


def _pad_2d4(values: np.ndarray, max_len: int, pad_value: float) -> Tuple[np.ndarray, np.ndarray]:
    trimmed = values[:max_len]
    padded = np.full((max_len, 4), pad_value, dtype=np.float32)
    padded[: len(trimmed)] = trimmed.astype(np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    mask[: len(trimmed)] = 1.0
    return padded, mask


def _slab_pairwise_dist(
    u: np.ndarray,
    v: np.ndarray,
    z_norm: np.ndarray,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    n_vec: np.ndarray,
    t: float,
) -> np.ndarray:
    n_atoms = u.shape[0]
    du0 = u[None, :] - u[:, None]
    dv0 = v[None, :] - v[:, None]
    best = None
    best_vec = None
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            du = du0 - m
            dv = dv0 - n
            delta_par = du[..., None] * a_hat[None, None, :] + dv[..., None] * b_hat[None, None, :]
            dist2 = np.sum(delta_par**2, axis=-1)
            if best is None:
                best = dist2
                best_vec = delta_par
            else:
                use = dist2 < best
                best = np.where(use, dist2, best)
                if best_vec is not None:
                    best_vec = np.where(use[..., None], delta_par, best_vec)
    if best_vec is None:
        best_vec = np.zeros((n_atoms, n_atoms, 3), dtype=np.float32)
    dz = (z_norm[None, :] - z_norm[:, None]) * float(t)
    r_ij = best_vec + dz[..., None] * n_vec[None, None, :]
    dist = np.linalg.norm(r_ij, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return dist.astype(np.float32)


def _build_slab_knn(
    u: np.ndarray,
    v: np.ndarray,
    z_norm: np.ndarray,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    n_vec: np.ndarray,
    t: float,
    max_atoms: int,
    k_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_atoms = u.shape[0]
    k = max(1, min(k_neighbors, max_atoms))
    nbr_idx = np.zeros((max_atoms, k), dtype=np.int64)
    nbr_dist = np.full((max_atoms, k), np.inf, dtype=np.float32)
    nbr_mask = np.zeros((max_atoms, k), dtype=np.float32)
    if n_atoms == 0:
        return nbr_idx, nbr_dist, nbr_mask
    dist = _slab_pairwise_dist(u, v, z_norm, a_hat, b_hat, n_vec, t)
    for i in range(n_atoms):
        row = dist[i]
        order = np.argsort(row)
        finite = np.isfinite(row[order])
        picks = order[finite][:k]
        if picks.size == 0:
            continue
        nbr_idx[i, : picks.size] = picks
        nbr_dist[i, : picks.size] = row[picks]
        nbr_mask[i, : picks.size] = 1.0
    return nbr_idx, nbr_dist, nbr_mask


def _lattice_to_gram6(lattice: np.ndarray) -> np.ndarray:
    gram = lattice.T @ lattice
    return np.array([gram[0, 0], gram[1, 1], gram[2, 2], gram[0, 1], gram[0, 2], gram[1, 2]], dtype=np.float32)


def row_to_tokens(
    cif_str: str,
    max_atoms: int,
    pad_value: float,
    g_scale: float,
    niggli_reduce: bool,
    preprocess_v3: bool,
    preprocess_cfg: PreprocessConfig,
    cache_neighbors: bool,
    neighbor_k: int,
) -> Optional[Dict[str, np.ndarray]]:
    structure = Structure.from_str(cif_str, fmt="cif")
    num_atoms = len(structure)
    if num_atoms > max_atoms:
        return None
    if niggli_reduce:
        structure = structure.get_reduced_structure("niggli")

    atomic_numbers = np.asarray([site.specie.number for site in structure], dtype=np.int64)
    frac_coords = np.asarray(structure.frac_coords, dtype=np.float32)
    lattice = np.asarray(structure.lattice.matrix, dtype=np.float32)
    pos_cart = np.asarray(structure.cart_coords, dtype=np.float64)

    padded_numbers, mask_numbers = _pad_1d(atomic_numbers, max_atoms, pad_value)
    padded_coords, mask_coords = _pad_2d(frac_coords, max_atoms, pad_value)
    atom_mask = np.minimum(mask_numbers, mask_coords)

    gram6 = _lattice_to_gram6(lattice) / g_scale
    payload = {
        "z": padded_numbers,
        "f": padded_coords,
        "atom_mask": atom_mask,
        "lattice": lattice,
        "gram6": gram6,
    }

    if preprocess_v3 and num_atoms > 0:
        pre = preprocess_cartesian(lattice.astype(np.float64), pos_cart, atomic_numbers, preprocess_cfg)
        uvz = np.stack([pre["u"], pre["v"], pre["z_norm"]], axis=-1)
        padded_z, _ = _pad_1d(pre["Z"], max_atoms, pad_value)
        padded_uvz, _ = _pad_2d(uvz, max_atoms, pad_value)
        padded_uv_angle, _ = _pad_2d4(pre["uv_angle"], max_atoms, pad_value)
        padded_u, _ = _pad_1d(pre["u"], max_atoms, pad_value)
        padded_v, _ = _pad_1d(pre["v"], max_atoms, pad_value)
        padded_z_norm, _ = _pad_1d(pre["z_norm"], max_atoms, pad_value)
        padded_order, _ = _pad_1d(pre["order_idx"], max_atoms, -1)
        payload.update(
            {
                "z_canon": padded_z,
                "uvz": padded_uvz,
                "uv_angle": padded_uv_angle,
                "u": padded_u,
                "v": padded_v,
                "z_norm": padded_z_norm,
                "t": pre["t"],
                "a_hat": pre["a_hat"],
                "b_hat": pre["b_hat"],
                "n": pre["n"],
                "lattice_param": pre["lattice_param"],
                "counts_vector": pre["counts_vector"],
                "order_idx": padded_order,
            }
        )
        if cache_neighbors:
            order_idx = pre["order_idx"].astype(np.int64)
            inv = np.empty_like(order_idx)
            inv[order_idx] = np.arange(order_idx.size)
            u_unsorted = pre["u"][inv]
            v_unsorted = pre["v"][inv]
            z_norm_unsorted = pre["z_norm"][inv]
            nbr_idx, nbr_dist, nbr_mask = _build_slab_knn(
                u_unsorted,
                v_unsorted,
                z_norm_unsorted,
                pre["a_hat"],
                pre["b_hat"],
                pre["n"],
                float(pre["t"]),
                max_atoms=max_atoms,
                k_neighbors=neighbor_k,
            )
            payload.update(
                {
                    "nbr_idx": nbr_idx,
                    "nbr_dist": nbr_dist,
                    "nbr_mask": nbr_mask,
                }
            )

    return payload


def build_dataset(
    csv_path: Path,
    max_atoms: int,
    pad_value: float,
    limit: Optional[int],
    g_scale: float,
    niggli_reduce: bool,
    preprocess_v3: bool,
    preprocess_cfg: PreprocessConfig,
    cache_neighbors: bool,
    neighbor_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    z_list: List[np.ndarray] = []
    f_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    lattice_list: List[np.ndarray] = []
    gram_list: List[np.ndarray] = []
    z_canon_list: List[np.ndarray] = []
    uvz_list: List[np.ndarray] = []
    uv_angle_list: List[np.ndarray] = []
    u_list: List[np.ndarray] = []
    v_list: List[np.ndarray] = []
    z_norm_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    a_hat_list: List[np.ndarray] = []
    b_hat_list: List[np.ndarray] = []
    n_list: List[np.ndarray] = []
    lattice_param_list: List[np.ndarray] = []
    counts_list: List[np.ndarray] = []
    order_list: List[np.ndarray] = []
    nbr_idx_list: List[np.ndarray] = []
    nbr_dist_list: List[np.ndarray] = []
    nbr_mask_list: List[np.ndarray] = []
    material_ids: List[str] = []

    for row in df.itertuples(index=False):
        cif = getattr(row, "cif", None)
        if not isinstance(cif, str) or not cif.strip():
            continue
        try:
            result = row_to_tokens(
                cif,
                max_atoms=max_atoms,
                pad_value=pad_value,
                g_scale=g_scale,
                niggli_reduce=niggli_reduce,
                preprocess_v3=preprocess_v3,
                preprocess_cfg=preprocess_cfg,
                cache_neighbors=cache_neighbors,
                neighbor_k=neighbor_k,
            )
        except Exception:
            continue
        if result is None:
            continue
        z_list.append(result["z"])
        f_list.append(result["f"])
        mask_list.append(result["atom_mask"])
        lattice_list.append(result["lattice"])
        gram_list.append(result["gram6"])
        if preprocess_v3 and "z_canon" in result:
            z_canon_list.append(result["z_canon"])
            uvz_list.append(result["uvz"])
            uv_angle_list.append(result["uv_angle"])
            u_list.append(result["u"])
            v_list.append(result["v"])
            z_norm_list.append(result["z_norm"])
            t_list.append(result["t"])
            a_hat_list.append(result["a_hat"])
            b_hat_list.append(result["b_hat"])
            n_list.append(result["n"])
            lattice_param_list.append(result["lattice_param"])
            counts_list.append(result["counts_vector"])
            order_list.append(result["order_idx"])
            if cache_neighbors and "nbr_idx" in result:
                nbr_idx_list.append(result["nbr_idx"])
                nbr_dist_list.append(result["nbr_dist"])
                nbr_mask_list.append(result["nbr_mask"])
        material_ids.append(str(getattr(row, "material_id", "")))

    if not z_list:
        z = np.zeros((0, max_atoms), dtype=np.int64)
        f = np.zeros((0, max_atoms, 3), dtype=np.float32)
        mask = np.zeros((0, max_atoms), dtype=np.float32)
        lattice = np.zeros((0, 3, 3), dtype=np.float32)
        gram6 = np.zeros((0, 6), dtype=np.float32)
        extras: Dict[str, np.ndarray] = {}
        return z, f, mask, lattice, gram6, [], extras

    z = np.stack(z_list, axis=0)
    f = np.stack(f_list, axis=0)
    mask = np.stack(mask_list, axis=0)
    lattice = np.stack(lattice_list, axis=0)
    gram6 = np.stack(gram_list, axis=0)
    extras: Dict[str, np.ndarray] = {}
    if preprocess_v3 and z_canon_list:
        extras = {
            "z_canon": np.stack(z_canon_list, axis=0),
            "uvz": np.stack(uvz_list, axis=0),
            "uv_angle": np.stack(uv_angle_list, axis=0),
            "u": np.stack(u_list, axis=0),
            "v": np.stack(v_list, axis=0),
            "z_norm": np.stack(z_norm_list, axis=0),
            "t": np.stack(t_list, axis=0),
            "a_hat": np.stack(a_hat_list, axis=0),
            "b_hat": np.stack(b_hat_list, axis=0),
            "n": np.stack(n_list, axis=0),
            "lattice_param": np.stack(lattice_param_list, axis=0),
            "counts_vector": np.stack(counts_list, axis=0),
            "order_idx": np.stack(order_list, axis=0),
        }
        if cache_neighbors and nbr_idx_list:
            extras.update(
                {
                    "nbr_idx": np.stack(nbr_idx_list, axis=0),
                    "nbr_dist": np.stack(nbr_dist_list, axis=0),
                    "nbr_mask": np.stack(nbr_mask_list, axis=0),
                }
            )
    return z, f, mask, lattice, gram6, material_ids, extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert C2DB CSV to token npz (Z/F/L/g).")
    parser.add_argument("--csv", type=Path, default=Path("data/C2DB/c2db_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/C2DB/ache/c2db_tokens.npz"))
    parser.add_argument("--max-atoms", type=int, default=24)
    parser.add_argument("--pad-value", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--g-scale", type=float, default=100.0)
    parser.add_argument("--niggli-reduce", action="store_true", help="Apply Niggli reduction to lattices.")
    parser.add_argument(
        "--preprocess-v3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run A++ v3 preprocessing and store canonical slab features.",
    )
    parser.add_argument("--eps-area", type=float, default=1e-6)
    parser.add_argument("--eps-inv", type=float, default=1e-12)
    parser.add_argument("--round-prec", type=float, default=1e-6)
    parser.add_argument("--z-norm-clip", type=float, default=1.5)
    parser.add_argument("--cache-neighbors", action="store_true", help="Cache slab kNN neighbor graph in npz.")
    parser.add_argument("--neighbor-k", type=int, default=16, help="k for cached neighbor graph.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_cfg = PreprocessConfig(
        eps_area=args.eps_area,
        eps_inv=args.eps_inv,
        round_prec=args.round_prec,
        z_norm_clip=args.z_norm_clip,
    )
    z, f, mask, lattice, gram6, material_ids, extras = build_dataset(
        csv_path=args.csv,
        max_atoms=args.max_atoms,
        pad_value=args.pad_value,
        limit=args.limit,
        g_scale=args.g_scale,
        niggli_reduce=args.niggli_reduce,
        preprocess_v3=args.preprocess_v3,
        preprocess_cfg=preprocess_cfg,
        cache_neighbors=args.cache_neighbors,
        neighbor_k=args.neighbor_k,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "z": z,
        "f": f,
        "atom_mask": mask,
        "lattice": lattice,
        "gram6": gram6,
        "material_id": np.array(material_ids),
        "max_atoms": np.asarray(args.max_atoms, dtype=np.int64),
        "g_scale": np.asarray(args.g_scale, dtype=np.float32),
        "preprocess_v3": np.asarray(int(args.preprocess_v3), dtype=np.int64),
        "preprocess_version": np.array("A++_v3"),
    }
    if args.preprocess_v3 and extras:
        payload.update(
            {
                "eps_area": np.asarray(args.eps_area, dtype=np.float32),
                "eps_inv": np.asarray(args.eps_inv, dtype=np.float32),
                "round_prec": np.asarray(args.round_prec, dtype=np.float32),
                "z_norm_clip": np.asarray(args.z_norm_clip, dtype=np.float32),
            }
        )
        if args.cache_neighbors:
            payload["neighbor_k"] = np.asarray(args.neighbor_k, dtype=np.int64)
        if "lattice_param" in extras:
            lattice = extras["lattice_param"].astype(np.float32)
            lattice_mean = lattice.mean(axis=0)
            lattice_std = lattice.std(axis=0)
            lattice_std = np.maximum(lattice_std, 1e-6)
            payload["cond_lattice_mean"] = lattice_mean
            payload["cond_lattice_std"] = lattice_std
        if "t" in extras:
            t_vals = extras["t"].astype(np.float32)
            t_mean = float(np.mean(t_vals))
            t_std = float(np.std(t_vals))
            if t_std < 1e-6:
                t_std = 1e-6
            payload["cond_t_mean"] = np.asarray(t_mean, dtype=np.float32)
            payload["cond_t_std"] = np.asarray(t_std, dtype=np.float32)
        payload.update(extras)
    np.savez_compressed(args.out, **payload)
    print(f"Saved {len(z)} samples to {args.out}")


if __name__ == "__main__":
    main()
