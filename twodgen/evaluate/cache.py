from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from twodgen.common.geometry_np import choose_vacuum_axis, min_dist_and_shifts, thickness_vacuum


CACHE_VERSION = "eval_cache_v1"


def _default_cache_path(samples_path: Path) -> Path:
    return samples_path.parent / "eval_cache.npz"


def build_eval_cache(
    samples_path: Path,
    *,
    out_path: Optional[Path] = None,
    pbc_mask: Tuple[int, int, int] = (1, 1, 0),
    bond_cut: float = 3.0,
) -> Path:
    samples = np.load(samples_path)
    z = samples["z"]
    frac = samples["frac"]
    lattice = samples["lattice"]
    atom_mask = samples["atom_mask"]
    cross_vacuum_flag = np.zeros((z.shape[0],), dtype=np.int8)
    thickness = np.full((z.shape[0],), np.nan, dtype=np.float32)
    vacuum = np.full((z.shape[0],), np.nan, dtype=np.float32)

    for i in range(z.shape[0]):
        mask = (atom_mask[i] > 0.5) & (z[i] > 0)
        if not np.any(mask):
            continue
        frac_i = frac[i][mask]
        lattice_i = lattice[i]
        c_idx, c_len, _ = choose_vacuum_axis(lattice_i)
        thickness_i, vacuum_i = thickness_vacuum(frac_i[:, c_idx], c_len)
        thickness[i] = float(thickness_i)
        vacuum[i] = float(vacuum_i)
        if mask.sum() > 1:
            if int(pbc_mask[c_idx]) == 0:
                _, dist_3d, shifts_3d = min_dist_and_shifts(frac_i, lattice_i, pbc_mask=(1, 1, 1))
                edges = np.where(dist_3d < float(bond_cut))
                cross_vac = False
                for a, b in zip(edges[0].tolist(), edges[1].tolist()):
                    if a >= b:
                        continue
                    if abs(float(shifts_3d[a, b, c_idx])) > 0.0:
                        cross_vac = True
                        break
                cross_vacuum_flag[i] = int(cross_vac)

    stats = samples_path.stat()
    payload: Dict[str, np.ndarray] = {
        "cache_version": np.array(CACHE_VERSION),
        "cross_vacuum_flag": cross_vacuum_flag,
        "thickness": thickness,
        "vacuum": vacuum,
        "bond_cut": np.asarray(bond_cut, dtype=np.float32),
        "pbc_mask": np.asarray(pbc_mask, dtype=np.int8),
        "samples_mtime_ns": np.asarray(stats.st_mtime_ns, dtype=np.int64),
        "samples_size": np.asarray(stats.st_size, dtype=np.int64),
    }
    if "energy_mlip" in samples:
        payload["energy_mlip"] = np.asarray(samples["energy_mlip"])
    if "relaxed_flag" in samples:
        payload["relaxed_flag"] = np.asarray(samples["relaxed_flag"])

    out_path = out_path or _default_cache_path(Path(samples_path))
    np.savez_compressed(out_path, **payload)
    return out_path


def load_eval_cache(
    samples_path: Path,
    *,
    cache_path: Optional[Path] = None,
    pbc_mask: Tuple[int, int, int] = (1, 1, 0),
    bond_cut: float = 3.0,
) -> Dict[str, np.ndarray]:
    cache_path = cache_path or _default_cache_path(Path(samples_path))
    if cache_path.exists():
        cache = dict(np.load(cache_path))
        cache_version = str(cache.get("cache_version", "")).strip()
        if cache_version != CACHE_VERSION:
            cache = {}
        if cache:
            cached_cut = cache.get("bond_cut")
            cached_mask = cache.get("pbc_mask")
            if cached_cut is None or cached_mask is None:
                cache = {}
            else:
                if not np.isclose(float(cached_cut), float(bond_cut)):
                    cache = {}
                elif tuple(int(v) for v in cached_mask.tolist()) != tuple(int(v) for v in pbc_mask):
                    cache = {}
        if cache:
            stats = samples_path.stat()
            cached_mtime = cache.get("samples_mtime_ns")
            cached_size = cache.get("samples_size")
            if cached_mtime is None or cached_size is None:
                cache = {}
            else:
                if int(cached_mtime) != int(stats.st_mtime_ns) or int(cached_size) != int(stats.st_size):
                    cache = {}
        if cache:
            return cache
    build_eval_cache(samples_path, out_path=cache_path, pbc_mask=pbc_mask, bond_cut=bond_cut)
    return dict(np.load(cache_path))
