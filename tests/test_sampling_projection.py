import torch

from twodgen.common.projection import cond_gram, inplane_degenerate, post_step_project


def test_post_step_project_fixes_inplane_degeneracy_and_cond():
    # Construct a near-degenerate in-plane lattice (a,b almost collinear) with large cond.
    # Row-basis lattice.
    lattice = torch.tensor(
        [
            [
                [2.0, 0.0, 0.0],
                [1.999, 0.01, 0.0],
                [0.0, 0.0, 20.0],
            ]
        ],
        dtype=torch.float32,
    )
    before_cond = float(cond_gram(lattice, pbc_mask=(1, 1, 0)).item())
    before_degen = bool(
        inplane_degenerate(
            lattice,
            pbc_mask=(1, 1, 0),
            a_min=2.0,
            b_min=2.0,
            gamma_min=30.0,
            gamma_max=150.0,
            area_min=4.0,
        ).item()
    )
    assert before_degen

    lat2, stats = post_step_project(
        lattice,
        keys=("angle", "cond", "inplane"),
        pbc_mask=(1, 1, 0),
        angle_min=30.0,
        angle_max=150.0,
        cond_max=40.0,
        inplane_a_min=2.0,
        inplane_b_min=2.0,
        inplane_gamma_min=30.0,
        inplane_gamma_max=150.0,
        inplane_area_min=4.0,
        max_iters=2,
    )
    assert torch.isfinite(lat2).all()
    after_cond = float(cond_gram(lat2, pbc_mask=(1, 1, 0)).item())
    assert after_cond <= 40.0 + 1e-3
    after_degen = bool(
        inplane_degenerate(
            lat2,
            pbc_mask=(1, 1, 0),
            a_min=2.0,
            b_min=2.0,
            gamma_min=30.0,
            gamma_max=150.0,
            area_min=4.0,
        ).item()
    )
    assert not after_degen
    assert float(stats["delta_norm"].item()) >= 0.0
    assert float(stats["cond_before"].item()) == before_cond
