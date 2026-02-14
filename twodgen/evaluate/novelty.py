from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from twodgen.common.geometry_np import min_dist_and_shifts


FINGERPRINT_NAME_V0 = "comp_inplane_rdf_adf_v0"


def _parse_pbc_mask(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must be three comma-separated 0/1 values.")
    mask = tuple(int(p) for p in parts)
    if any(v not in (0, 1) for v in mask):
        raise ValueError("--pbc-mask values must be 0 or 1.")
    return mask  # type: ignore[return-value]


def _load_npz_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    z = np.asarray(data["z"])
    atom_mask = np.asarray(data["atom_mask"])
    lattice = np.asarray(data["lattice"])
    if "frac" in data:
        frac = np.asarray(data["frac"])
    elif "f" in data:
        frac = np.asarray(data["f"])
    else:
        raise KeyError(f"{path} must contain frac or f.")
    return z, frac, lattice, atom_mask


def _composition_feature(z_i: np.ndarray, num_elements: int = 118) -> np.ndarray:
    feat = np.zeros((num_elements,), dtype=float)
    if z_i.size == 0:
        return feat
    valid = (z_i > 0) & (z_i <= num_elements)
    if not np.any(valid):
        return feat
    values, counts = np.unique(z_i[valid].astype(int), return_counts=True)
    feat[values - 1] = counts.astype(float)
    feat_sum = feat.sum()
    if feat_sum > 0:
        feat /= feat_sum
    return feat


def _canonicalize_inplane(
    lattice: np.ndarray, frac: np.ndarray, pbc_mask: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], int]:
    periodic = [idx for idx, flag in enumerate(pbc_mask) if int(flag) == 1]
    if len(periodic) != 2:
        periodic = [0, 1]
    non_periodic = [idx for idx in range(3) if idx not in periodic]
    vac_axis = non_periodic[0] if non_periodic else 2
    # Put periodic axes first and sort by length to keep canonical in-plane order.
    lengths = np.linalg.norm(lattice[periodic], axis=1)
    order2 = np.argsort(lengths)
    periodic_sorted = [periodic[int(order2[0])], periodic[int(order2[1])]]
    order = periodic_sorted + [vac_axis]
    lattice_new = lattice[order]
    try:
        inv_new = np.linalg.inv(lattice_new)
        frac_new = frac @ lattice @ inv_new
        frac_new = frac_new - np.floor(frac_new)
    except np.linalg.LinAlgError:
        frac_new = frac
    if np.linalg.det(lattice_new) < 0:
        lattice_new[0] *= -1.0
    return lattice_new, frac_new, (0, 1), 2


def _inplane_feature(lattice: np.ndarray, inplane_axes: Tuple[int, int]) -> np.ndarray:
    a_vec = lattice[inplane_axes[0]]
    b_vec = lattice[inplane_axes[1]]
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))
    denom = max(a_len * b_len, 1e-12)
    cos_g = float(np.clip(np.dot(a_vec, b_vec) / denom, -1.0, 1.0))
    gamma = float(np.degrees(np.arccos(cos_g)))
    area = float(np.linalg.norm(np.cross(a_vec, b_vec)))
    return np.asarray([a_len, b_len, gamma / 180.0, area], dtype=float)


def _rdf_hist(
    frac_i: np.ndarray,
    lattice_i: np.ndarray,
    *,
    pbc_mask: Tuple[int, int, int],
    bins: int,
    r_max: float,
) -> np.ndarray:
    if frac_i.shape[0] <= 1:
        return np.zeros((bins,), dtype=float)
    _, dist, _ = min_dist_and_shifts(frac_i, lattice_i, pbc_mask=pbc_mask)
    tri = np.triu_indices(dist.shape[0], k=1)
    vals = dist[tri]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.zeros((bins,), dtype=float)
    hist, _ = np.histogram(vals, bins=bins, range=(0.0, float(r_max)), density=False)
    hist = hist.astype(float)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _adf_hist(
    frac_i: np.ndarray,
    lattice_i: np.ndarray,
    *,
    pbc_mask: Tuple[int, int, int],
    bins: int,
    k_neighbors: int,
) -> np.ndarray:
    n = int(frac_i.shape[0])
    if n < 3:
        return np.zeros((bins,), dtype=float)
    _, dist, shifts = min_dist_and_shifts(frac_i, lattice_i, pbc_mask=pbc_mask)
    angle_values: List[float] = []
    for center in range(n):
        d = dist[center].copy()
        d[center] = np.inf
        order = np.argsort(d)
        neigh = [idx for idx in order.tolist() if np.isfinite(d[idx])][: max(2, int(k_neighbors))]
        if len(neigh) < 2:
            continue
        vecs: List[np.ndarray] = []
        for nb in neigh:
            # dr(center, nb) = frac_center - frac_nb - shift -> vector center->nb is -dr.
            shift = shifts[center, nb]
            vec_frac = frac_i[nb] + shift - frac_i[center]
            vec_cart = vec_frac @ lattice_i
            norm = float(np.linalg.norm(vec_cart))
            if norm <= 1e-12 or not np.isfinite(norm):
                continue
            vecs.append(vec_cart / norm)
        if len(vecs) < 2:
            continue
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                cos_v = float(np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0))
                angle_values.append(float(np.degrees(np.arccos(cos_v))))
    if not angle_values:
        return np.zeros((bins,), dtype=float)
    hist, _ = np.histogram(np.asarray(angle_values, dtype=float), bins=bins, range=(0.0, 180.0))
    hist = hist.astype(float)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _fingerprint_one(
    z_i: np.ndarray,
    frac_i: np.ndarray,
    lattice_i: np.ndarray,
    *,
    pbc_mask: Tuple[int, int, int],
    rdf_bins: int,
    adf_bins: int,
    rdf_r_max: float,
    adf_k: int,
) -> np.ndarray:
    lattice_c, frac_c, inplane_axes, _ = _canonicalize_inplane(lattice_i, frac_i, pbc_mask)
    comp = _composition_feature(z_i)
    inplane = _inplane_feature(lattice_c, inplane_axes)
    rdf = _rdf_hist(
        frac_c,
        lattice_c,
        pbc_mask=pbc_mask,
        bins=int(rdf_bins),
        r_max=float(rdf_r_max),
    )
    adf = _adf_hist(
        frac_c,
        lattice_c,
        pbc_mask=pbc_mask,
        bins=int(adf_bins),
        k_neighbors=int(adf_k),
    )
    return np.concatenate([comp, inplane, rdf, adf], axis=0)


def _fingerprint_matrix(
    z: np.ndarray,
    frac: np.ndarray,
    lattice: np.ndarray,
    atom_mask: np.ndarray,
    *,
    pbc_mask: Tuple[int, int, int],
    rdf_bins: int,
    adf_bins: int,
    rdf_r_max: float,
    adf_k: int,
) -> np.ndarray:
    feats: List[np.ndarray] = []
    for i in range(z.shape[0]):
        mask = atom_mask[i] > 0.5
        z_i = z[i][mask].astype(int)
        frac_i = frac[i][mask]
        lat_i = lattice[i]
        feats.append(
            _fingerprint_one(
                z_i,
                frac_i,
                lat_i,
                pbc_mask=pbc_mask,
                rdf_bins=rdf_bins,
                adf_bins=adf_bins,
                rdf_r_max=rdf_r_max,
                adf_k=adf_k,
            )
        )
    if not feats:
        return np.zeros((0, 118 + 4 + rdf_bins + adf_bins), dtype=float)
    return np.stack(feats, axis=0)


def _nearest_distance(x: np.ndarray, ref: np.ndarray, chunk: int = 1024) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    if ref.shape[0] == 0:
        return np.full((x.shape[0],), float("inf"), dtype=float)
    out = np.full((x.shape[0],), float("inf"), dtype=float)
    for start in range(0, x.shape[0], chunk):
        end = min(start + chunk, x.shape[0])
        block = x[start:end]
        # (B,1,D) - (1,R,D) -> (B,R,D)
        diff = block[:, None, :] - ref[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        out[start:end] = np.min(dist, axis=1)
    return out


def _greedy_dedup(feat: np.ndarray, threshold: float) -> np.ndarray:
    if feat.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    keep = np.zeros((feat.shape[0],), dtype=bool)
    kept_idx: List[int] = []
    for i in range(feat.shape[0]):
        if not kept_idx:
            keep[i] = True
            kept_idx.append(i)
            continue
        ref = feat[np.asarray(kept_idx, dtype=int)]
        d = np.linalg.norm(ref - feat[i], axis=1)
        if np.all(d > float(threshold)):
            keep[i] = True
            kept_idx.append(i)
    return keep


def _update_metrics_summary(
    metrics_summary: Path,
    *,
    fingerprint_name: str,
    fingerprint_params: Dict[str, Any],
    novelty_mean: float,
    novelty_median: float,
    dedup_keep_rate: float,
) -> None:
    data = json.loads(metrics_summary.read_text(encoding="utf-8"))
    data["novelty_fingerprint"] = {
        "name": fingerprint_name,
        "params": fingerprint_params,
    }
    data["novelty_mean"] = float(novelty_mean)
    data["novelty_median"] = float(novelty_median)
    data["dedup_keep_rate"] = float(dedup_keep_rate)
    metrics_summary.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Novelty score and dedup for generated samples.")
    parser.add_argument("--samples", type=Path, required=True, help="Generated samples npz.")
    parser.add_argument("--train-npz", type=Path, required=True, help="Reference training npz.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pbc-mask", type=str, default="1,1,0")
    parser.add_argument("--rdf-bins", type=int, default=16)
    parser.add_argument("--adf-bins", type=int, default=12)
    parser.add_argument("--rdf-r-max", type=float, default=6.0)
    parser.add_argument("--adf-k", type=int, default=4)
    parser.add_argument("--dedup-threshold", type=float, default=0.12)
    parser.add_argument(
        "--metrics-summary",
        type=Path,
        default=None,
        help="Optional metrics_summary.json to inject fingerprint metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pbc_mask = _parse_pbc_mask(args.pbc_mask)
    fingerprint_params = {
        "pbc_mask": pbc_mask,
        "rdf_bins": int(args.rdf_bins),
        "adf_bins": int(args.adf_bins),
        "rdf_r_max": float(args.rdf_r_max),
        "adf_k": int(args.adf_k),
        "canonicalization": {
            "vacuum_axis_alignment": True,
            "cell_reduction": "inplane_length_sort",
            "atom_permutation_invariant": True,
        },
    }

    z_s, f_s, l_s, m_s = _load_npz_arrays(args.samples)
    z_t, f_t, l_t, m_t = _load_npz_arrays(args.train_npz)
    feat_s = _fingerprint_matrix(
        z_s,
        f_s,
        l_s,
        m_s,
        pbc_mask=pbc_mask,
        rdf_bins=args.rdf_bins,
        adf_bins=args.adf_bins,
        rdf_r_max=args.rdf_r_max,
        adf_k=args.adf_k,
    )
    feat_t = _fingerprint_matrix(
        z_t,
        f_t,
        l_t,
        m_t,
        pbc_mask=pbc_mask,
        rdf_bins=args.rdf_bins,
        adf_bins=args.adf_bins,
        rdf_r_max=args.rdf_r_max,
        adf_k=args.adf_k,
    )
    novelty = _nearest_distance(feat_s, feat_t)
    keep = _greedy_dedup(feat_s, threshold=float(args.dedup_threshold))

    rows: List[Dict[str, Any]] = []
    for i in range(feat_s.shape[0]):
        rows.append(
            {
                "id": int(i),
                "novelty_score": float(novelty[i]),
                "dedup_keep": bool(keep[i]),
            }
        )
    with (args.out_dir / "per_sample_novelty.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "fingerprint_name": FINGERPRINT_NAME_V0,
        "fingerprint_params": fingerprint_params,
        "total_samples": int(feat_s.shape[0]),
        "reference_samples": int(feat_t.shape[0]),
        "novelty": {
            "mean": float(np.mean(novelty)) if novelty.size else None,
            "median": float(np.median(novelty)) if novelty.size else None,
            "p90": float(np.percentile(novelty, 90.0)) if novelty.size else None,
            "max": float(np.max(novelty)) if novelty.size else None,
        },
        "dedup_threshold": float(args.dedup_threshold),
        "dedup_keep_count": int(np.sum(keep)),
        "dedup_keep_rate": float(np.mean(keep)) if keep.size else 0.0,
    }
    (args.out_dir / "novelty_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    if args.metrics_summary is not None and args.metrics_summary.exists():
        _update_metrics_summary(
            args.metrics_summary,
            fingerprint_name=FINGERPRINT_NAME_V0,
            fingerprint_params=fingerprint_params,
            novelty_mean=float(summary["novelty"]["mean"]) if summary["novelty"]["mean"] is not None else float("nan"),
            novelty_median=float(summary["novelty"]["median"]) if summary["novelty"]["median"] is not None else float("nan"),
            dedup_keep_rate=float(summary["dedup_keep_rate"]),
        )

    print(f"Saved novelty outputs to {args.out_dir}")


if __name__ == "__main__":
    main()

