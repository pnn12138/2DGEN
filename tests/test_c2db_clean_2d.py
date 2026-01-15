import numpy as np

from twodgen.data.clean_c2db_2d import _mic_dist_and_shifts, _thickness_vacuum


def test_thickness_vacuum_max_gap_method() -> None:
    # Three atoms in a 20Å vacuum axis; the maximum gap is the wrap-around gap.
    frac = np.array([0.1, 0.2, 0.21], dtype=float)
    thickness, vacuum = _thickness_vacuum(frac, c_len=20.0)
    assert abs(thickness - 2.2) < 1e-6
    assert abs(vacuum - 17.8) < 1e-6


def test_cross_vacuum_bond_detection_via_shifts() -> None:
    lattice = np.diag([3.0, 3.0, 20.0]).astype(float)
    frac = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.9]], dtype=float)

    dist_3d, shifts_3d = _mic_dist_and_shifts(frac, lattice, pbc_mask=(1, 1, 1))
    assert abs(float(dist_3d[0, 1]) - 4.0) < 1e-6
    assert shifts_3d[0, 1, 2] != 0.0

    dist_slab, shifts_slab = _mic_dist_and_shifts(frac, lattice, pbc_mask=(1, 1, 0))
    assert abs(float(dist_slab[0, 1]) - 16.0) < 1e-6
    assert shifts_slab[0, 1, 2] == 0.0

