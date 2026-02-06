import json
from pathlib import Path

import numpy as np

from twodgen.evaluate.eval_samples import _eval_samples


def test_energy_skipped_reason_geom_fail(tmp_path: Path):
    # One sample, but invalid due to bad volume -> should skip energy with geom_fail.
    samples = {
        "z": np.array([[1, 1, 0]], dtype=np.int64),
        "atom_mask": np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        "frac": np.array([[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]]], dtype=np.float32),
        # Extremely large volume triggers bad_volume when v_min/v_max are set.
        "lattice": np.array([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]], dtype=np.float32),
        "energy_mlip": np.array([-10.0], dtype=np.float32),
        "relaxed_flag": np.array([1], dtype=np.int64),
    }
    per_sample, tier0, _, _ = _eval_samples(
        samples,
        v_min=1.0,
        v_max=10.0,
        min_dist_cut=1.5,
        bond_cut=3.0,
        dup_eps=1e-3,
        pbc_mask=(1, 1, 0),
        vacuum_min=15.0,
        atomic_ref_map=None,
        formation_energy_threshold=0.0,
        cond_max=None,
    )
    row = per_sample[0]
    assert row["success_geom"] is False
    assert row["energy_skipped_reason"] == "geom_fail"
    assert row["success_energy"] is None
    assert tier0["energy_skipped_geom_rate"] == 1.0


def test_energy_missing_classified(tmp_path: Path):
    samples = {
        "z": np.array([[1, 1, 1]], dtype=np.int64),
        "atom_mask": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        # Space atoms far enough to avoid collision under min_dist_cut=1.5 with a=b=3.0
        "frac": np.array([[[0.1, 0.1, 0.5], [0.7, 0.1, 0.5], [0.1, 0.7, 0.5]]], dtype=np.float32),
        # Use a=b=4.0 so MIC distances for 0.4 fractional separation exceed 1.5A.
        "lattice": np.array([[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 20.0]]], dtype=np.float32),
        "energy_mlip": np.array([np.nan], dtype=np.float32),
        "relaxed_flag": np.array([0], dtype=np.int64),
    }
    per_sample, tier0, _, _ = _eval_samples(
        samples,
        v_min=None,
        v_max=None,
        min_dist_cut=1.5,
        bond_cut=3.0,
        dup_eps=1e-3,
        pbc_mask=(1, 1, 0),
        vacuum_min=15.0,
        atomic_ref_map=None,
        formation_energy_threshold=0.0,
        cond_max=1e6,
    )
    row = per_sample[0]
    assert row["success_geom"] is True
    assert row["energy_skipped_reason"] == "mlip_unavailable"
    assert row["fail_reason_energy"] == "missing"
    assert row["success_energy"] is None
    assert tier0["energy_skipped_mlip_rate"] == 1.0
