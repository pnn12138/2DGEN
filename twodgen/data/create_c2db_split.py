from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SCHEMA_VERSION = "c2db_split_v1"


def _quantile_bins(values: np.ndarray, bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0 or bins <= 1:
        return np.array([float("-inf"), float("inf")], dtype=float)
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(finite, qs).astype(float)
    # Ensure monotonic edges to avoid empty bins due to constant values.
    edges[0] = float("-inf")
    edges[-1] = float("inf")
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # edges: (-inf, e1, ..., inf) => bins are [e0,e1), ..., [e_{k-1}, e_k)
    return np.digitize(values, edges[1:-1], right=False).astype(int)


def _normalized_hist(values: np.ndarray, num_bins: int) -> np.ndarray:
    counts = np.bincount(values.astype(int), minlength=num_bins).astype(float)
    total = max(float(counts.sum()), 1.0)
    return counts / total


def _max_abs_diff(p: np.ndarray, q: np.ndarray) -> float:
    n = max(p.size, q.size)
    p2 = np.pad(p, (0, max(0, n - p.size)))
    q2 = np.pad(q, (0, max(0, n - q.size)))
    return float(np.max(np.abs(p2 - q2)))


def create_split(
    *,
    npz_path: Path,
    heldout_fraction: float,
    seed: int,
    t_bins: int,
) -> Dict[str, Any]:
    data = np.load(npz_path)
    if "atom_mask" not in data and "z" not in data:
        raise ValueError("Expected token cache to contain at least atom_mask or z fields.")

    # Accept both the full-prepared token cache (counts_vector/t present) and
    # minimal caches where only z/atom_mask are available.
    atom_mask = data["atom_mask"].astype(np.float32) if "atom_mask" in data else None
    z = data["z"].astype(np.int64) if "z" in data else None
    if atom_mask is None and z is not None:
        atom_mask = (z > 0).astype(np.float32)
    if atom_mask is None:
        raise ValueError("Token cache missing atom_mask (and cannot infer it without z).")

    total = int(atom_mask.shape[0])
    if total == 0:
        keys = sorted(list(data.files))
        raise ValueError(
            "Empty token cache (0 samples). "
            f"npz={npz_path} keys={keys}. "
            "Re-generate the token cache first (prepare_c2db_tokens) and re-run create_c2db_split."
        )

    # counts_vector: (N, E) element counts. If absent, compute from (z, atom_mask).
    if "counts_vector" in data:
        counts_vector = data["counts_vector"].astype(np.int64)
    else:
        if z is None:
            raise ValueError("Token cache missing counts_vector and z; cannot compute element counts.")
        # Default to 118 elements (Z=1..118) to match the rest of the pipeline.
        num_elements = 118
        counts_vector = np.zeros((total, num_elements), dtype=np.int64)
        valid = (atom_mask > 0.5) & (z > 0)
        for i in range(total):
            zs = z[i][valid[i]].astype(int)
            if zs.size == 0:
                continue
            idx = zs - 1
            idx = idx[(idx >= 0) & (idx < num_elements)]
            if idx.size:
                np.add.at(counts_vector[i], idx, 1)

    n_atoms = atom_mask.sum(axis=1).round().astype(int)

    top_elem = np.argmax(counts_vector, axis=1).astype(int)  # 0..(E-1)
    has_atoms = counts_vector.sum(axis=1) > 0
    top_elem = np.where(has_atoms, top_elem, -1)

    # Thickness stratification: prefer normalized thickness t if present.
    t = data["t"].astype(np.float32).reshape(-1) if "t" in data else None
    if t is None:
        t_bin = np.zeros((total,), dtype=int)
        t_edges = np.array([float("-inf"), float("inf")], dtype=float)
    else:
        t_edges = _quantile_bins(t, bins=t_bins)
        t_bin = _bin_index(t, t_edges)

    rng = np.random.default_rng(seed)
    strata: Dict[Tuple[int, int, int], List[int]] = {}
    for idx in range(total):
        key = (int(n_atoms[idx]), int(t_bin[idx]), int(top_elem[idx]))
        strata.setdefault(key, []).append(idx)

    target_heldout = int(round(total * heldout_fraction))
    heldout: List[int] = []
    for _, indices in strata.items():
        n = len(indices)
        if n <= 1:
            continue
        k = int(np.floor(n * heldout_fraction))
        if k <= 0 and n >= 10 and heldout_fraction > 0:
            k = 1
        if k <= 0:
            continue
        picked = rng.choice(indices, size=min(k, n), replace=False).tolist()
        heldout.extend(picked)

    heldout_set = set(heldout)
    remaining = [i for i in range(total) if i not in heldout_set]
    if len(heldout) < target_heldout:
        need = target_heldout - len(heldout)
        add = rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist()
        heldout.extend(add)
    elif len(heldout) > target_heldout:
        heldout = rng.choice(heldout, size=target_heldout, replace=False).tolist()

    heldout = sorted(set(int(x) for x in heldout))
    heldout_set = set(heldout)
    train = [i for i in range(total) if i not in heldout_set]

    # Distribution checks (coarse): normalized hist max-abs-diff.
    max_atoms = int(n_atoms.max()) if n_atoms.size else 0
    n_atoms_bins = max_atoms + 1
    p_n = _normalized_hist(n_atoms.astype(int), n_atoms_bins)
    p_n_train = _normalized_hist(n_atoms[train].astype(int), n_atoms_bins)
    p_n_held = _normalized_hist(n_atoms[heldout].astype(int), n_atoms_bins)

    # Element hist by presence (counts over all atoms) to approximate global element distribution.
    elem_total = counts_vector.sum(axis=0).astype(float)
    elem_train = counts_vector[train].sum(axis=0).astype(float)
    elem_held = counts_vector[heldout].sum(axis=0).astype(float)
    elem_total /= max(elem_total.sum(), 1.0)
    elem_train /= max(elem_train.sum(), 1.0)
    elem_held /= max(elem_held.sum(), 1.0)

    t_bins_used = int(t_bin.max() + 1) if t_bin.size else 1
    p_t = _normalized_hist(t_bin, t_bins_used)
    p_t_train = _normalized_hist(t_bin[train], t_bins_used)
    p_t_held = _normalized_hist(t_bin[heldout], t_bins_used)

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "source_npz": str(npz_path),
        "seed": int(seed),
        "heldout_fraction": float(heldout_fraction),
        "t_bins": int(t_bins),
        "t_bin_edges": [float(x) for x in t_edges.tolist()],
        "split": {
            "train_indices": train,
            "heldout_indices": heldout,
        },
        "summary": {
            "total": int(total),
            "train": int(len(train)),
            "heldout": int(len(heldout)),
        },
        "distribution_checks": {
            "n_atoms_max_abs_diff_train": _max_abs_diff(p_n, p_n_train),
            "n_atoms_max_abs_diff_heldout": _max_abs_diff(p_n, p_n_held),
            "elements_max_abs_diff_train": float(np.max(np.abs(elem_total - elem_train))),
            "elements_max_abs_diff_heldout": float(np.max(np.abs(elem_total - elem_held))),
            "t_bin_max_abs_diff_train": _max_abs_diff(p_t, p_t_train),
            "t_bin_max_abs_diff_heldout": _max_abs_diff(p_t, p_t_held),
        },
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create explicit train/held-out split for C2DB token cache.")
    parser.add_argument("--npz", type=Path, required=True, help="Token cache npz (from prepare_c2db_tokens.py).")
    parser.add_argument("--out", type=Path, required=True, help="Output split json.")
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-bins", type=int, default=10, help="Quantile bins for slab thickness stratification.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if not (0.0 < args.heldout_fraction < 1.0):
        raise ValueError("--heldout-fraction must be in (0, 1).")
    payload = create_split(
        npz_path=args.npz,
        heldout_fraction=float(args.heldout_fraction),
        seed=int(args.seed),
        t_bins=int(args.t_bins),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    checks = payload.get("distribution_checks", {})
    print(f"Saved split to {args.out}")
    print(f"[check] n_atoms max_abs_diff train/heldout: {checks.get('n_atoms_max_abs_diff_train'):.4f}/{checks.get('n_atoms_max_abs_diff_heldout'):.4f}")
    print(f"[check] elements max_abs_diff train/heldout: {checks.get('elements_max_abs_diff_train'):.4f}/{checks.get('elements_max_abs_diff_heldout'):.4f}")
    print(f"[check] t_bin max_abs_diff train/heldout: {checks.get('t_bin_max_abs_diff_train'):.4f}/{checks.get('t_bin_max_abs_diff_heldout'):.4f}")


if __name__ == "__main__":
    main()
