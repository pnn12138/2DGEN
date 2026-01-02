from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from twodgen.data.preprocess import PreprocessConfig, preprocess_cartesian


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


def _lattice_to_gram6(lattice: np.ndarray) -> np.ndarray:
    # Convention: lattice basis vectors are stored in rows (cart = frac @ lattice).
    gram = lattice @ lattice.T
    return np.array([gram[0, 0], gram[1, 1], gram[2, 2], gram[0, 1], gram[0, 2], gram[1, 2]], dtype=np.float32)


def row_to_tokens(
    cif_str: str,
    max_atoms: int,
    pad_value: float,
    g_scale: float,
    niggli_reduce: bool,
    preprocess_v3: bool,
    preprocess_cfg: PreprocessConfig,
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
    return z, f, mask, lattice, gram6, material_ids, extras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert C2DB CSV to token npz (Z/F/L/g).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
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
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "z": z,
        "f": f,
        "atom_mask": mask,
        "lattice": lattice,
        "gram6": gram6,
        "gram6_convention": np.array("row_lattice"),
        "gram6_version": np.asarray(2, dtype=np.int64),
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
