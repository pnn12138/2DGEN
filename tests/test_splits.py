import json
from pathlib import Path

import numpy as np

from twodgen.data.create_c2db_split import create_split
from twodgen.data.splits import load_c2db_split, select_split_indices, validate_split_indices


def _write_dummy_npz(path: Path, n: int = 200, e: int = 8) -> None:
    rng = np.random.default_rng(0)
    atom_mask = np.zeros((n, 6), dtype=np.float32)
    counts = np.zeros((n, e), dtype=np.int64)
    t = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        n_atoms = int(rng.integers(1, 6))
        atom_mask[i, :n_atoms] = 1.0
        elems = rng.integers(0, e, size=n_atoms)
        for z in elems:
            counts[i, z] += 1
        t[i] = float(rng.normal(loc=8.0 + 0.2 * n_atoms, scale=0.5))
    np.savez_compressed(path, atom_mask=atom_mask, counts_vector=counts, t=t)


def test_create_and_load_split(tmp_path: Path) -> None:
    npz_path = tmp_path / "dummy.npz"
    _write_dummy_npz(npz_path, n=200, e=10)
    payload = create_split(npz_path=npz_path, heldout_fraction=0.2, seed=123, t_bins=5)
    out = tmp_path / "split.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    split = load_c2db_split(out)
    train = select_split_indices(split, "train")
    held = select_split_indices(split, "heldout")
    all_idx = select_split_indices(split, "all")
    assert len(set(train).intersection(set(held))) == 0
    assert sorted(all_idx) == list(range(200))
    validate_split_indices(train, total=200)
    validate_split_indices(held, total=200)

