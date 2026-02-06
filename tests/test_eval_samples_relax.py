import numpy as np

from twodgen.evaluate.eval_samples import _eval_samples


def test_eval_samples_relax_fields_and_energy_gate() -> None:
    z = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int64)
    frac = np.array(
        [
            [[0.1, 0.1, 0.1], [0.7, 0.1, 0.1], [0.1, 0.7, 0.1]],
            [[0.2, 0.2, 0.1], [0.8, 0.2, 0.1], [0.2, 0.8, 0.1]],
        ],
        dtype=np.float32,
    )
    lattice = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
    # Ensure in-plane metrics pass (a,b >= 2.0 and non-degenerate in-plane area)
    lattice[:, 0, 0] = 3.0
    lattice[:, 1, 1] = 3.0
    lattice[:, 2, 2] = 20.0
    atom_mask = np.ones((2, 3), dtype=np.float32)
    samples = {
        "z": z,
        "frac": frac,
        "lattice": lattice,
        "atom_mask": atom_mask,
        "energy_mlip": np.array([np.nan, -3.0], dtype=np.float32),
        "relaxed_flag": np.array([0, 1], dtype=np.int64),
        "min_dist_relax": np.array([np.nan, 2.0], dtype=np.float32),
    }
    per_sample, tier0, _, _ = _eval_samples(
        samples,
        v_min=None,
        v_max=None,
        min_dist_cut=0.1,
        bond_cut=3.0,
        dup_eps=1e-3,
        pbc_mask=(1, 1, 0),
        vacuum_min=None,
        atomic_ref_map=[0.0, 0.0],
        formation_energy_threshold=0.0,
        success_top_k=0,
        target_spacegroup=None,
        spacegroup_symprec=1e-2,
    )
    assert len(per_sample) == 2
    assert per_sample[0]["energy_mlip"] is None
    assert per_sample[1]["energy_mlip"] == -3.0
    assert per_sample[0]["relaxed_flag"] == 0
    assert per_sample[1]["relaxed_flag"] == 1
    assert per_sample[0]["min_dist_relax"] is None
    assert per_sample[1]["min_dist_relax"] == 2.0
    assert tier0["relaxed_rate"] == 0.5
    assert tier0["min_dist_relax"]["count"] == 1
    assert tier0["success_rate"] == 1.0
    assert tier0["success_energy_rate"] == 1.0
