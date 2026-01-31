from __future__ import annotations

from typing import Tuple

import numpy as np


def choose_vacuum_axis(lattice: np.ndarray) -> tuple[int, float, np.ndarray]:
    lengths = np.linalg.norm(lattice, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        return 2, float("nan"), lengths
    c_idx = int(np.argmax(lengths))
    return c_idx, float(lengths[c_idx]), lengths


def thickness_vacuum(frac_c: np.ndarray, c_len: float) -> Tuple[float, float]:
    if frac_c.size == 0:
        return float("nan"), float("nan")
    coords = np.sort(frac_c)
    if coords.size == 1:
        thickness = 0.0
        return thickness, c_len - thickness
    gaps = np.diff(coords, axis=0).flatten().tolist()
    gaps.append(1.0 - (coords[-1] - coords[0]))
    max_gap = max(gaps)
    thickness = (1.0 - max_gap) * c_len
    vacuum = c_len - thickness
    return float(thickness), float(vacuum)


def min_dist_and_shifts(
    frac: np.ndarray, lattice: np.ndarray, pbc_mask: Tuple[int, int, int]
) -> Tuple[float, np.ndarray, np.ndarray]:
    if frac.size == 0:
        return float("inf"), np.zeros((0, 0)), np.zeros((0, 0, 3))
    df = frac[:, None, :] - frac[None, :, :]

    shifts_1d = (-1.0, 0.0, 1.0)
    zeros_1d = (0.0,)
    components = [
        shifts_1d if pbc_mask[0] == 1 else zeros_1d,
        shifts_1d if pbc_mask[1] == 1 else zeros_1d,
        shifts_1d if pbc_mask[2] == 1 else zeros_1d,
    ]
    shifts_all = np.asarray(list(_cartesian_product(components)), dtype=float)

    df_shifted = df[:, :, None, :] - shifts_all[None, None, :, :]
    dr = df_shifted @ lattice
    dist_all = np.linalg.norm(dr, axis=-1)
    best_idx = np.argmin(dist_all, axis=-1)
    dist = np.take_along_axis(dist_all, best_idx[:, :, None], axis=-1)[:, :, 0]
    shifts = shifts_all[best_idx]

    np.fill_diagonal(dist, np.inf)
    min_dist = float(np.min(dist)) if dist.size > 0 else float("inf")
    return min_dist, dist, shifts


def _cartesian_product(components):
    if len(components) != 3:
        raise ValueError("expected 3 components for cartesian product")
    for a in components[0]:
        for b in components[1]:
            for c in components[2]:
                yield (a, b, c)


__all__ = ["choose_vacuum_axis", "thickness_vacuum", "min_dist_and_shifts"]
