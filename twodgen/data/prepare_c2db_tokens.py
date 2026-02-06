from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure

try:  # Optional dependency: spacegroup metadata uses spglib via pymatgen.
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    SpacegroupAnalyzer = None  # type: ignore

from twodgen.common.geometry_np import choose_vacuum_axis, min_dist_and_shifts, thickness_vacuum
from twodgen.data.preprocess import PreprocessConfig, preprocess_cartesian
from twodgen.data.utils import pad_1d, pad_2d, pad_2d4, wrap01_array


def _cross_vacuum_bond(frac: np.ndarray, lattice: np.ndarray, bond_cut: float) -> bool:
    c_idx, _, _ = choose_vacuum_axis(lattice)
    if frac.shape[0] < 2:
        return False
    _, dist_3d, shifts_3d = min_dist_and_shifts(frac.astype(float), lattice.astype(float), pbc_mask=(1, 1, 1))
    edges = np.where(dist_3d < float(bond_cut))
    if edges[0].size == 0:
        return False
    for a, b in zip(edges[0].tolist(), edges[1].tolist()):
        if a >= b:
            continue
        if abs(float(shifts_3d[a, b, c_idx])) > 0.0:
            return True
    return False


def _lattice_to_gram6(lattice: np.ndarray) -> np.ndarray:
    # Convention: lattice basis vectors are stored in rows (cart = frac @ lattice).
    gram = lattice @ lattice.T
    return np.array([gram[0, 0], gram[1, 1], gram[2, 2], gram[0, 1], gram[0, 2], gram[1, 2]], dtype=np.float32)


def _invert_order(order_idx: np.ndarray) -> np.ndarray:
    order_inv = np.full_like(order_idx, -1)
    if order_idx.size == 0:
        return order_inv
    order_inv[order_idx] = np.arange(order_idx.size, dtype=order_idx.dtype)
    return order_inv


def _counts_vector(atomic_numbers: np.ndarray, max_atomic_number: int = 118) -> np.ndarray:
    counts = np.zeros((max_atomic_number,), dtype=np.int64)
    for z_val in atomic_numbers.astype(int).tolist():
        if 1 <= z_val <= max_atomic_number:
            counts[z_val - 1] += 1
    return counts


def _spacegroup_info(structure: Structure) -> Tuple[int, str]:
    if SpacegroupAnalyzer is None:
        return -1, ""
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-2, angle_tolerance=5.0)
        number = int(analyzer.get_space_group_number())
        symbol = str(analyzer.get_space_group_symbol())
        return number, symbol
    except Exception:
        return -1, ""


def row_to_tokens(
    cif_str: str,
    max_atoms: int,
    pad_value: float,
    g_scale: float,
    niggli_reduce: bool,
    preprocess_cfg: PreprocessConfig,
    min_dist_cut: float,
    pbc_mask: Tuple[int, int, int],
    vacuum_min: float,
    bond_cut: float,
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
    spacegroup_number, spacegroup_symbol = _spacegroup_info(structure)

    padded_numbers, mask_numbers = pad_1d(atomic_numbers, max_atoms, pad_value)
    padded_coords, mask_coords = pad_2d(frac_coords, max_atoms, pad_value)
    atom_mask = np.minimum(mask_numbers, mask_coords)

    gram6 = _lattice_to_gram6(lattice) / g_scale
    if num_atoms >= 2:
        _, dist, _ = min_dist_and_shifts(
            frac_coords.astype(float),
            lattice.astype(float),
            pbc_mask=pbc_mask,
        )
        min_dist = float(np.min(dist)) if dist.size else float("inf")
    else:
        min_dist = float("inf")
    collision_risk = float(min_dist < float(min_dist_cut))
    c_idx, c_len, _ = choose_vacuum_axis(lattice)
    frac_c = wrap01_array(frac_coords[:, c_idx])
    thickness, vacuum = thickness_vacuum(frac_c, c_len)
    low_vacuum_risk = bool(np.isfinite(vacuum) and vacuum < float(vacuum_min))
    cross_vacuum_bond = _cross_vacuum_bond(frac_coords, lattice, bond_cut=bond_cut)
    payload = {
        "z": padded_numbers,
        "f": padded_coords,
        "atom_mask": atom_mask,
        "lattice": lattice,
        "gram6": gram6,
        "min_dist": np.asarray(min_dist, dtype=np.float32),
        "collision_risk": np.asarray(collision_risk, dtype=np.float32),
        "counts_vector": _counts_vector(atomic_numbers).astype(np.int64),
        "slab_thickness": np.asarray(thickness, dtype=np.float32),
        "slab_vacuum": np.asarray(vacuum, dtype=np.float32),
        "low_vacuum_risk": np.asarray(int(low_vacuum_risk), dtype=np.int64),
        "cross_vacuum_bond": np.asarray(int(cross_vacuum_bond), dtype=np.int64),
        "spacegroup_number": np.asarray(spacegroup_number, dtype=np.int64),
        # np.unicode_ was removed in NumPy 2.0; np.str_ preserves a stable unicode dtype.
        "spacegroup_symbol": np.asarray(spacegroup_symbol, dtype=np.str_),
    }

    if num_atoms > 0:
        pre = preprocess_cartesian(lattice.astype(np.float64), pos_cart, atomic_numbers, preprocess_cfg)
        order_idx = pre["order_idx"]
        order_inv = _invert_order(order_idx)

        canon_numbers = atomic_numbers[order_idx]
        canon_coords = frac_coords[order_idx]
        padded_z_canon, mask_z_canon = pad_1d(canon_numbers, max_atoms, pad_value)
        padded_f_canon, mask_f_canon = pad_2d(canon_coords, max_atoms, pad_value)
        atom_mask_canon = np.minimum(mask_z_canon, mask_f_canon)

        lattice_canon = pre["lattice_canon"].astype(np.float32)
        try:
            inv_lattice_canon = np.linalg.inv(lattice_canon.astype(np.float64))
        except np.linalg.LinAlgError:
            return None
        frac_canon = pos_cart @ inv_lattice_canon
        frac_canon = wrap01_array(frac_canon)
        u_shift = float(pre["u_shift"])
        v_shift = float(pre["v_shift"])
        frac_canon[:, 0] = wrap01_array(frac_canon[:, 0] - u_shift)
        frac_canon[:, 1] = wrap01_array(frac_canon[:, 1] - v_shift)
        frac_canon = frac_canon[order_idx]
        padded_frac_canon, mask_frac_canon = pad_2d(frac_canon.astype(np.float32), max_atoms, pad_value)
        atom_mask_canon = np.minimum(atom_mask_canon, mask_frac_canon)
        gram6_canon = _lattice_to_gram6(lattice_canon) / g_scale

        uvz = np.stack([pre["u"], pre["v"], pre["z_norm"]], axis=-1)
        padded_z, _ = pad_1d(pre["Z"], max_atoms, pad_value)
        padded_uvz, _ = pad_2d(uvz, max_atoms, pad_value)
        padded_uv_angle, _ = pad_2d4(pre["uv_angle"], max_atoms, pad_value)
        padded_u, _ = pad_1d(pre["u"], max_atoms, pad_value)
        padded_v, _ = pad_1d(pre["v"], max_atoms, pad_value)
        padded_z_norm, _ = pad_1d(pre["z_norm"], max_atoms, pad_value)
        padded_order, _ = pad_1d(pre["order_idx"], max_atoms, -1)
        padded_order_inv, _ = pad_1d(order_inv, max_atoms, -1)
        payload.update(
            {
                "z_canon": padded_z,
                "f_canon": padded_frac_canon,
                "atom_mask_canon": atom_mask_canon,
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
                "order_inv": padded_order_inv,
                "lattice_canon": lattice_canon,
                "gram6_canon": gram6_canon.astype(np.float32),
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
    preprocess_cfg: PreprocessConfig,
    min_dist_cut: float,
    pbc_mask: Tuple[int, int, int],
    vacuum_min: float,
    bond_cut: float,
    verbose: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, np.ndarray],
    Dict[str, int],
]:
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    stats = {
        "total_rows": 0,
        "skipped_empty": 0,
        "skipped_parse": 0,
        "skipped_too_many_atoms": 0,
    }

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
    f_canon_list: List[np.ndarray] = []
    atom_mask_canon_list: List[np.ndarray] = []
    order_inv_list: List[np.ndarray] = []
    lattice_canon_list: List[np.ndarray] = []
    gram6_canon_list: List[np.ndarray] = []
    material_ids: List[str] = []
    min_dist_list: List[np.ndarray] = []
    collision_risk_list: List[np.ndarray] = []
    thickness_list: List[np.ndarray] = []
    vacuum_list: List[np.ndarray] = []
    low_vacuum_list: List[np.ndarray] = []
    cross_vacuum_list: List[np.ndarray] = []
    spacegroup_number_list: List[np.ndarray] = []
    spacegroup_symbol_list: List[np.ndarray] = []

    error_examples: List[str] = []
    for row in df.itertuples(index=False):
        stats["total_rows"] += 1
        cif = getattr(row, "cif", None)
        if not isinstance(cif, str) or not cif.strip():
            stats["skipped_empty"] += 1
            continue
        try:
            result = row_to_tokens(
                cif,
                max_atoms=max_atoms,
                pad_value=pad_value,
                g_scale=g_scale,
                niggli_reduce=niggli_reduce,
                preprocess_cfg=preprocess_cfg,
                min_dist_cut=min_dist_cut,
                pbc_mask=pbc_mask,
                vacuum_min=vacuum_min,
                bond_cut=bond_cut,
            )
        except Exception as exc:
            stats["skipped_parse"] += 1
            if verbose and len(error_examples) < 5:
                error_examples.append(f"parse_error: {exc!r}")
            continue
        if result is None:
            stats["skipped_too_many_atoms"] += 1
            continue
        z_list.append(result["z"])
        f_list.append(result["f"])
        mask_list.append(result["atom_mask"])
        lattice_list.append(result["lattice"])
        gram_list.append(result["gram6"])
        min_dist_list.append(result["min_dist"])
        collision_risk_list.append(result["collision_risk"])
        counts_list.append(result["counts_vector"])
        thickness_list.append(result["slab_thickness"])
        vacuum_list.append(result["slab_vacuum"])
        low_vacuum_list.append(result["low_vacuum_risk"])
        cross_vacuum_list.append(result["cross_vacuum_bond"])
        spacegroup_number_list.append(result["spacegroup_number"])
        spacegroup_symbol_list.append(result["spacegroup_symbol"])
        if "z_canon" in result:
            z_canon_list.append(result["z_canon"])
            if "f_canon" in result:
                f_canon_list.append(result["f_canon"])
            if "atom_mask_canon" in result:
                atom_mask_canon_list.append(result["atom_mask_canon"])
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
            order_list.append(result["order_idx"])
            if "order_inv" in result:
                order_inv_list.append(result["order_inv"])
            if "lattice_canon" in result:
                lattice_canon_list.append(result["lattice_canon"])
            if "gram6_canon" in result:
                gram6_canon_list.append(result["gram6_canon"])
        material_ids.append(str(getattr(row, "material_id", "")))

    if not z_list:
        z = np.zeros((0, max_atoms), dtype=np.int64)
        f = np.zeros((0, max_atoms, 3), dtype=np.float32)
        mask = np.zeros((0, max_atoms), dtype=np.float32)
        lattice = np.zeros((0, 3, 3), dtype=np.float32)
        gram6 = np.zeros((0, 6), dtype=np.float32)
        extras: Dict[str, np.ndarray] = {}
        return z, f, mask, lattice, gram6, [], extras, stats

    z = np.stack(z_list, axis=0)
    f = np.stack(f_list, axis=0)
    mask = np.stack(mask_list, axis=0)
    lattice = np.stack(lattice_list, axis=0)
    gram6 = np.stack(gram_list, axis=0)
    extras: Dict[str, np.ndarray] = {
        "min_dist": np.stack(min_dist_list, axis=0).astype(np.float32),
        "collision_risk": np.stack(collision_risk_list, axis=0).astype(np.float32),
        "counts_vector": np.stack(counts_list, axis=0).astype(np.int64),
        "slab_thickness": np.stack(thickness_list, axis=0).astype(np.float32),
        "slab_vacuum": np.stack(vacuum_list, axis=0).astype(np.float32),
        "low_vacuum_risk": np.stack(low_vacuum_list, axis=0).astype(np.int64),
        "cross_vacuum_bond": np.stack(cross_vacuum_list, axis=0).astype(np.int64),
        "spacegroup_number": np.stack(spacegroup_number_list, axis=0).astype(np.int64),
        "spacegroup_symbol": np.stack(spacegroup_symbol_list, axis=0),
    }
    if z_canon_list:
        extras.update(
            {
            "z_canon": np.stack(z_canon_list, axis=0),
            "f_canon": np.stack(f_canon_list, axis=0) if f_canon_list else None,
            "atom_mask_canon": np.stack(atom_mask_canon_list, axis=0) if atom_mask_canon_list else None,
            "lattice_canon": np.stack(lattice_canon_list, axis=0) if lattice_canon_list else None,
            "gram6_canon": np.stack(gram6_canon_list, axis=0) if gram6_canon_list else None,
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
            "order_idx": np.stack(order_list, axis=0),
            "order_inv": np.stack(order_inv_list, axis=0) if order_inv_list else None,
            }
        )
        extras = {k: v for k, v in extras.items() if v is not None}
    if error_examples:
        stats["error_examples"] = error_examples
    return z, f, mask, lattice, gram6, material_ids, extras, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert C2DB CSV to token npz (Z/F/L/g).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--csv", type=Path, default=Path("data/C2DB/c2db_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/C2DB/cache/c2db_tokens.npz"))
    parser.add_argument("--max-atoms", type=int, default=24)
    parser.add_argument("--pad-value", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--g-scale", type=float, default=100.0)
    parser.add_argument("--niggli-reduce", action="store_true", help="Apply Niggli reduction to lattices.")
    parser.add_argument("--eps-area", type=float, default=1e-6)
    parser.add_argument("--eps-inv", type=float, default=1e-12)
    parser.add_argument("--round-prec", type=float, default=1e-6)
    parser.add_argument("--z-norm-clip", type=float, default=1.5)
    parser.add_argument(
        "--min-dist-cut",
        type=float,
        default=1.5,
        help="Collision risk threshold (Angstrom) stored in the cache.",
    )
    parser.add_argument(
        "--pbc-mask",
        type=str,
        default="1,1,0",
        help="Periodic dimensions for min_dist computation, e.g. 1,1,0 for 2D slab MIC.",
    )
    parser.add_argument(
        "--vacuum-min",
        type=float,
        default=15.0,
        help="Minimum vacuum thickness (Angstrom) used to tag low_vacuum_risk in the cache.",
    )
    parser.add_argument(
        "--bond-cut",
        type=float,
        default=3.0,
        help="Bond cutoff (Angstrom) used to detect cross_vacuum_bond under 3D PBC.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print sample errors during preprocessing.")
    return parser.parse_args()

def _parse_pbc_mask(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must have three comma-separated values, e.g. 1,1,0")
    mask = tuple(int(p) for p in parts)
    if any(p not in (0, 1) for p in mask):
        raise ValueError("--pbc-mask values must be 0 or 1")
    return mask  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    pbc_mask = _parse_pbc_mask(args.pbc_mask)
    preprocess_cfg = PreprocessConfig(
        eps_area=args.eps_area,
        eps_inv=args.eps_inv,
        round_prec=args.round_prec,
        z_norm_clip=args.z_norm_clip,
    )
    z, f, mask, lattice, gram6, material_ids, extras, stats = build_dataset(
        csv_path=args.csv,
        max_atoms=args.max_atoms,
        pad_value=args.pad_value,
        limit=args.limit,
        g_scale=args.g_scale,
        niggli_reduce=args.niggli_reduce,
        preprocess_cfg=preprocess_cfg,
        min_dist_cut=args.min_dist_cut,
        pbc_mask=pbc_mask,
        vacuum_min=args.vacuum_min,
        bond_cut=args.bond_cut,
        verbose=args.verbose,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    has_canon = extras.get("f_canon") is not None and extras.get("gram6_canon") is not None
    coord_frame_value = "canon" if has_canon else "raw"
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
        "preprocess_v3": np.asarray(1, dtype=np.int64),
        "preprocess_version": np.array("A++_v3"),
        "schema_version": np.array("v4"),
        "coord_frame": np.array(coord_frame_value),
        "min_dist_cut": np.asarray(args.min_dist_cut, dtype=np.float32),
        "min_dist_pbc_mask": np.asarray(pbc_mask, dtype=np.int64),
        "vacuum_min": np.asarray(args.vacuum_min, dtype=np.float32),
        "bond_cut": np.asarray(args.bond_cut, dtype=np.float32),
    }
    if extras:
        payload.update(
            {
                "eps_area": np.asarray(args.eps_area, dtype=np.float32),
                "eps_inv": np.asarray(args.eps_inv, dtype=np.float32),
                "round_prec": np.asarray(args.round_prec, dtype=np.float32),
                "z_norm_clip": np.asarray(args.z_norm_clip, dtype=np.float32),
            }
        )
        payload.update(extras)
    np.savez_compressed(args.out, **payload)
    stats["saved_samples"] = int(z.shape[0])
    stats_path = args.out.parent / "preprocess_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"Saved preprocessing stats to {stats_path}")
    print(f"Saved {len(z)} samples to {args.out}")


if __name__ == "__main__":
    main()
