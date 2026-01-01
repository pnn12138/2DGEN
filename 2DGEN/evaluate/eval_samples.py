from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _parse_pbc_mask(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must have three comma-separated values, e.g. 1,1,0")
    mask = tuple(int(p) for p in parts)
    if any(p not in (0, 1) for p in mask):
        raise ValueError("--pbc-mask values must be 0 or 1")
    return mask  # type: ignore[return-value]


def _load_npz_stats(npz_path: Path) -> Optional[Tuple[float, float]]:
    data = np.load(npz_path)
    lattice = data["lattice"] if "lattice" in data else None
    if lattice is None:
        return None
    vols = np.abs(np.linalg.det(lattice))
    v_min = float(np.percentile(vols, 1.0))
    v_max = float(np.percentile(vols, 99.0))
    return v_min, v_max


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def _min_dist_and_shifts(
    frac: np.ndarray, lattice: np.ndarray, pbc_mask: Tuple[int, int, int]
) -> Tuple[float, np.ndarray, np.ndarray]:
    df = frac[:, None, :] - frac[None, :, :]
    pbc = np.asarray(pbc_mask, dtype=float).reshape((1, 1, 3))
    shifts = np.round(df) * pbc
    df_mic = df - shifts
    dr = df_mic @ lattice
    dist = np.linalg.norm(dr, axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_dist = float(np.min(dist)) if dist.size > 0 else float("inf")
    return min_dist, dist, shifts


def _thickness_vacuum(frac: np.ndarray, c_len: float) -> Tuple[float, float]:
    if frac.size == 0:
        return float("nan"), float("nan")
    coords = np.sort(frac)
    if coords.size == 1:
        thickness = 0.0
        return thickness, c_len - thickness
    gaps = np.diff(coords, axis=0).flatten().tolist()
    gaps.append(1.0 - (coords[-1] - coords[0]))
    max_gap = max(gaps)
    thickness = (1.0 - max_gap) * c_len
    vacuum = c_len - thickness
    return float(thickness), float(vacuum)


def _gcc_ratio(n_atoms: int, edges: List[Tuple[int, int]]) -> float:
    if n_atoms == 0:
        return 0.0
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    visited = [False] * n_atoms
    max_size = 0
    for i in range(n_atoms):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        size = 0
        while stack:
            cur = stack.pop()
            size += 1
            for nxt in adj[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        max_size = max(max_size, size)
    return float(max_size / n_atoms)


def _eval_samples(
    samples: Dict[str, np.ndarray],
    v_min: Optional[float],
    v_max: Optional[float],
    min_dist_cut: float,
    bond_cut: float,
    dup_eps: float,
    pbc_mask: Tuple[int, int, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    z = samples["z"]
    frac = samples["frac"]
    lattice = samples["lattice"]
    atom_mask = samples["atom_mask"]

    per_sample: List[Dict[str, Any]] = []
    fail_counts: Dict[str, int] = {}
    elem_counts: Dict[str, int] = {}

    min_dists: List[float] = []
    volumes: List[float] = []
    conds: List[float] = []
    n_atoms_list: List[int] = []
    angles_alpha: List[float] = []
    angles_beta: List[float] = []
    angles_gamma: List[float] = []
    angle_out_flags: List[int] = []

    thicknesses: List[float] = []
    vacuums: List[float] = []
    cross_vacuum_flags: List[int] = []
    gcc_ratios: List[float] = []
    anisotropies: List[float] = []

    valid_flags: List[int] = []
    valid_2d_flags: List[int] = []

    for i in range(z.shape[0]):
        mask = atom_mask[i] > 0.5
        z_i = z[i][mask].astype(int)
        frac_i = frac[i][mask]
        lattice_i = lattice[i]
        n_atoms = int(mask.sum())

        reasons: List[str] = []
        if n_atoms == 0:
            reasons.append("empty_atoms")
        if n_atoms < 3:
            reasons.append("low_atoms")

        vol = float(abs(np.linalg.det(lattice_i)))
        if v_min is not None and v_max is not None:
            if vol < v_min or vol > v_max:
                reasons.append("bad_volume")

        gram = lattice_i.T @ lattice_i
        eigvals = np.linalg.eigvalsh(gram)
        if not np.all(np.isfinite(eigvals)) or np.any(eigvals <= 0.0):
            reasons.append("non_spd")
            cond = float("inf")
        else:
            cond = float(eigvals.max() / max(eigvals.min(), 1e-12))

        if n_atoms > 0:
            min_dist, dist, shifts = _min_dist_and_shifts(frac_i, lattice_i, pbc_mask=pbc_mask)
            min_dists.append(min_dist)
            if min_dist < min_dist_cut:
                reasons.append("collision")

            quant = np.round(frac_i / dup_eps)
            uniq = np.unique(quant, axis=0)
            dup_ratio = 1.0 - (len(uniq) / n_atoms)
            if dup_ratio > 0.2:
                reasons.append("duplicate_coord")
        else:
            min_dist = float("inf")
            dist = np.zeros((0, 0))
            shifts = np.zeros((0, 0, 3))
            dup_ratio = float("nan")

        lengths = np.linalg.norm(lattice_i, axis=1)
        if np.all(lengths > 0):
            a_vec, b_vec, c_vec = lattice_i
            cos_alpha = float(np.dot(b_vec, c_vec) / (np.linalg.norm(b_vec) * np.linalg.norm(c_vec)))
            cos_beta = float(np.dot(a_vec, c_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(c_vec)))
            cos_gamma = float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))
            cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
            cos_beta = float(np.clip(cos_beta, -1.0, 1.0))
            cos_gamma = float(np.clip(cos_gamma, -1.0, 1.0))
            alpha = float(np.degrees(np.arccos(cos_alpha)))
            beta = float(np.degrees(np.arccos(cos_beta)))
            gamma = float(np.degrees(np.arccos(cos_gamma)))
            angles_alpha.append(alpha)
            angles_beta.append(beta)
            angles_gamma.append(gamma)
            angle_out = int((alpha < 30.0 or alpha > 150.0) or (beta < 30.0 or beta > 150.0) or (gamma < 30.0 or gamma > 150.0))
            angle_out_flags.append(angle_out)
            if angle_out:
                reasons.append("angle_out_of_range")
        c_idx = int(np.argmax(lengths))
        c_len = float(lengths[c_idx])
        if n_atoms > 0:
            thickness, vacuum = _thickness_vacuum(frac_i[:, c_idx], c_len)
        else:
            thickness, vacuum = float("nan"), float("nan")

        cross_vacuum = False
        edges: List[Tuple[int, int]] = []
        if n_atoms > 1:
            dist_3d, shifts_3d = None, None
            if pbc_mask[c_idx] == 0:
                _, dist_3d, shifts_3d = _min_dist_and_shifts(frac_i, lattice_i, pbc_mask=(1, 1, 1))
            for a in range(n_atoms):
                for b in range(a + 1, n_atoms):
                    if dist[a, b] < bond_cut:
                        edges.append((a, b))
                    if dist_3d is not None and dist_3d[a, b] < bond_cut:
                        n_c = shifts_3d[a, b, c_idx]
                        if n_c != 0:
                            cross_vacuum = True
        gcc_ratio = _gcc_ratio(n_atoms, edges)

        anisotropy = float(c_len / max(np.mean([l for j, l in enumerate(lengths) if j != c_idx]), 1e-8))

        valid = len(reasons) == 0
        valid_2d = valid and (not cross_vacuum)

        for reason in reasons:
            fail_counts[reason] = fail_counts.get(reason, 0) + 1

        if valid:
            for z_val in z_i.tolist():
                if z_val > 0:
                    key = str(z_val)
                    elem_counts[key] = elem_counts.get(key, 0) + 1

        volumes.append(vol)
        conds.append(cond)
        n_atoms_list.append(n_atoms)
        thicknesses.append(thickness)
        vacuums.append(vacuum)
        cross_vacuum_flags.append(int(cross_vacuum))
        gcc_ratios.append(gcc_ratio)
        anisotropies.append(anisotropy)
        valid_flags.append(int(valid))
        valid_2d_flags.append(int(valid_2d))

        per_sample.append(
            {
                "id": int(i),
                "n_atoms": n_atoms,
                "volume": vol,
                "cond": cond,
                "min_dist": min_dist,
                "dup_ratio": dup_ratio,
                "thickness": thickness,
                "vacuum": vacuum,
                "cross_vacuum_bond": cross_vacuum,
                "gcc_ratio": gcc_ratio,
                "anisotropy": anisotropy,
                "valid": valid,
                "valid_2d": valid_2d,
                "fail_reason": "+".join(reasons) if reasons else "",
            }
        )

    tier0 = {
        "valid_rate": float(np.mean(valid_flags)) if valid_flags else 0.0,
        "fail_reason_counts": fail_counts,
        "min_dist": _summary_stats(min_dists),
        "volume": _summary_stats(volumes),
        "cond": _summary_stats(conds),
        "spd_rate": float(np.mean([c < float("inf") for c in conds])) if conds else 0.0,
        "angle_alpha": _summary_stats(angles_alpha),
        "angle_beta": _summary_stats(angles_beta),
        "angle_gamma": _summary_stats(angles_gamma),
        "angle_out_of_range_rate": float(np.mean(angle_out_flags)) if angle_out_flags else 0.0,
        "n_atoms": _summary_stats(n_atoms_list),
        "element_counts": elem_counts,
        "total_samples": int(z.shape[0]),
    }
    tier1 = {
        "valid_2d_rate": float(np.mean(valid_2d_flags)) if valid_2d_flags else 0.0,
        "thickness": _summary_stats(thicknesses),
        "vacuum": _summary_stats(vacuums),
        "cross_vacuum_rate": float(np.mean(cross_vacuum_flags)) if cross_vacuum_flags else 0.0,
        "gcc_ratio": _summary_stats(gcc_ratios),
        "anisotropy": _summary_stats(anisotropies),
        "total_samples": int(z.shape[0]),
    }
    return per_sample, tier0, tier1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate token-based samples (Tier-0/1).")
    parser.add_argument("--samples", type=Path, default=None, help="Path to samples.npz")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for eval artifacts.")
    parser.add_argument("--stats-npz", type=Path, default=None, help="NPZ for volume bounds (p1/p99).")
    parser.add_argument("--min-dist", type=float, default=1.5)
    parser.add_argument("--bond-cut", type=float, default=3.0)
    parser.add_argument("--dup-eps", type=float, default=1e-3)
    parser.add_argument("--v-min", type=float, default=None)
    parser.add_argument("--v-max", type=float, default=None)
    parser.add_argument(
        "--pbc-mask",
        type=str,
        default="1,1,0",
        help="Comma-separated PBC mask for MIC distance, e.g. 1,1,0 for slab.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a small internal sanity check and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pbc_mask = _parse_pbc_mask(args.pbc_mask)
    if args.self_check:
        lattice = np.eye(3, dtype=float)
        frac = np.array([[0.1, 0.1, 0.1], [0.9, 0.1, 0.9]], dtype=float)
        d_3d, _, shifts_3d = _min_dist_and_shifts(frac, lattice, pbc_mask=(1, 1, 1))
        d_slab, _, shifts_slab = _min_dist_and_shifts(frac, lattice, pbc_mask=(1, 1, 0))
        assert abs(d_3d - (0.2**2 + 0.0**2 + 0.2**2) ** 0.5) < 1e-6, d_3d
        assert abs(d_slab - (0.2**2 + 0.0**2 + 0.8**2) ** 0.5) < 1e-6, d_slab
        assert np.all(shifts_slab[..., 2] == 0.0)
        print("self-check passed")
        return

    if args.samples is None:
        raise ValueError("--samples is required unless --self-check is set.")
    samples = np.load(args.samples)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.samples.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    v_min = args.v_min
    v_max = args.v_max
    if args.stats_npz is not None:
        stats = _load_npz_stats(args.stats_npz)
        if stats is not None:
            v_min, v_max = stats

    per_sample, tier0, tier1 = _eval_samples(
        samples,
        v_min=v_min,
        v_max=v_max,
        min_dist_cut=args.min_dist,
        bond_cut=args.bond_cut,
        dup_eps=args.dup_eps,
        pbc_mask=pbc_mask,
    )

    with (out_dir / "per_sample.jsonl").open("w", encoding="utf-8") as f:
        for row in per_sample:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    with (out_dir / "tier0_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(tier0, f, indent=2, ensure_ascii=True)
    with (out_dir / "tier1_2d_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(tier1, f, indent=2, ensure_ascii=True)

    print(f"Saved eval outputs to {out_dir}")


if __name__ == "__main__":
    main()
