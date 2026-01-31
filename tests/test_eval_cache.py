from __future__ import annotations

from pathlib import Path

import numpy as np

from twodgen.evaluate.cache import build_eval_cache, load_eval_cache


def test_eval_cache_build_and_load(tmp_path: Path) -> None:
    lattice = np.diag([5.0, 5.0, 10.0]).astype(np.float32)
    frac = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.9]], dtype=np.float32)
    samples_path = tmp_path / "samples.npz"

    np.savez_compressed(
        samples_path,
        z=np.array([[1, 1]], dtype=np.int64),
        frac=frac[None, ...],
        lattice=lattice[None, ...],
        atom_mask=np.array([[1.0, 1.0]], dtype=np.float32),
        energy_mlip=np.array([-1.23], dtype=np.float32),
        relaxed_flag=np.array([1], dtype=np.int8),
    )

    cache_path = build_eval_cache(samples_path, bond_cut=3.0)
    cache = load_eval_cache(samples_path)

    assert cache_path.exists()
    assert int(cache["cross_vacuum_flag"][0]) == 1
    assert np.isfinite(cache["thickness"][0])
    assert np.isfinite(cache["vacuum"][0])
    assert np.isclose(float(cache["energy_mlip"][0]), -1.23)
    assert int(cache["relaxed_flag"][0]) == 1
