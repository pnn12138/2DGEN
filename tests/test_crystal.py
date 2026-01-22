import torch

from twodgen.common.crystal import build_knn, cholesky6_to_lattice, frac_mic_dist


def test_frac_mic_dist_slab_vs_3d() -> None:
    lattice = torch.eye(3).unsqueeze(0)
    frac = torch.tensor([[[0.1, 0.1, 0.1], [0.9, 0.1, 0.9]]], dtype=torch.float32)
    mask = torch.ones(1, 2, dtype=torch.float32)

    dist_3d = frac_mic_dist(frac, lattice, mask, pbc_mask=(1, 1, 1))
    dist_slab = frac_mic_dist(frac, lattice, mask, pbc_mask=(1, 1, 0))
    expected_3d = (0.2**2 + 0.0**2 + 0.2**2) ** 0.5
    expected_slab = (0.2**2 + 0.0**2 + 0.8**2) ** 0.5

    assert torch.isfinite(dist_3d[0, 0, 1])
    assert abs(dist_3d[0, 0, 1].item() - expected_3d) < 1e-6
    assert abs(dist_slab[0, 0, 1].item() - expected_slab) < 1e-6


def test_build_knn_masks_infinite_rows() -> None:
    dist = torch.tensor(
        [
            [
                [float("inf"), 1.0, 2.0],
                [1.0, float("inf"), 3.0],
                [float("inf"), float("inf"), float("inf")],
            ]
        ]
    )
    idx, mask = build_knn(dist, k=2)
    assert idx.shape == (1, 3, 2)
    assert mask.shape == (1, 3, 2)
    assert mask[0, 2].sum().item() == 0


def test_cholesky6_to_lattice_supports_vector_bounds() -> None:
    y = torch.zeros((1, 6), dtype=torch.float32)
    y[0, :3] = torch.tensor([0.0, 1.0, 2.0])
    lattice = cholesky6_to_lattice(y, log_min=(0.5, 0.5, 0.5), log_max=(1.5, 1.5, 1.5))
    diag = torch.diagonal(lattice[0], dim1=-2, dim2=-1)
    expected = torch.exp(torch.tensor([0.5, 1.0, 1.5]))
    assert torch.allclose(diag, expected, atol=1e-6)
