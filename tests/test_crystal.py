import torch

from twodgen.common.crystal import (
    build_knn,
    cholesky6_to_lattice,
    frac_mic_dist,
    gram6_to_lattice,
    gram6_to_gram_matrix,
    gram_matrix_to_gram6,
    project_gram_cond_spd,
)


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


def test_project_gram_cond_spd_reduces_condition() -> None:
    eigs = torch.tensor([[1e-4, 1.0, 100.0]], dtype=torch.float32)
    gram = torch.diag_embed(eigs)
    gram6 = gram_matrix_to_gram6(gram)
    projected = project_gram_cond_spd(gram6, kappa_max=50.0, eps=1e-6, mode="log")
    gram_proj = gram6_to_gram_matrix(projected)
    eigvals = torch.linalg.eigvalsh(gram_proj)
    cond = eigvals.max(dim=-1).values / eigvals.min(dim=-1).values.clamp_min(1e-8)
    assert torch.all(cond <= 50.0 + 1e-2)


def test_gram6_roundtrip_has_gradients() -> None:
    torch.manual_seed(0)
    a = torch.randn(2, 3, 3, dtype=torch.float32, requires_grad=True)
    gram = a @ a.transpose(-1, -2) + 1e-3 * torch.eye(3)
    gram6 = gram_matrix_to_gram6(gram)
    lattice = gram6_to_lattice(gram6)
    gram6_round = gram_matrix_to_gram6(lattice @ lattice.transpose(-1, -2))
    loss = gram6_round.sum()
    loss.backward()
    assert a.grad is not None
    assert torch.isfinite(a.grad).all()
    assert a.grad.norm().item() > 0
