from __future__ import annotations

import torch

from twodgen.common.crystal import lattice_to_gram6
from twodgen.model.atom_denoiser import AtomDenoiser, AtomDenoiserConfig
from twodgen.model.tail_adapters import EgNNTailAdapter


def test_symmetry_loss_returns_scalar() -> None:
    denoiser = AtomDenoiser(AtomDenoiserConfig())
    lattice = torch.diag(torch.tensor([5.0, 5.0, 20.0])).unsqueeze(0)
    gram6 = lattice_to_gram6(lattice)
    frac = torch.tensor([[[0.1, 0.1, 0.1], [0.6, 0.6, 0.1]]], dtype=torch.float32)
    atom_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    z = torch.tensor([[1, 6]], dtype=torch.long)
    spacegroup_number = torch.tensor([1], dtype=torch.long)

    loss, rate = denoiser._symmetry_residual_loss(
        frac, gram6, atom_mask, z, spacegroup_number
    )

    assert loss.ndim == 0
    assert rate.ndim == 0
    assert 0.0 <= float(loss) <= 1.0
    assert 0.0 <= float(rate) <= 1.0


def test_egnn_tail_adapter_masks_atoms() -> None:
    adapter = EgNNTailAdapter(z_embed_dim=4, hidden_dim=8, pbc_mask=(1, 1, 0), init_scale=0.1)
    z_emb = torch.randn(1, 2, 4)
    frac = torch.tensor([[[0.1, 0.1, 0.1], [0.6, 0.6, 0.1]]], dtype=torch.float32)
    lattice = torch.diag(torch.tensor([5.0, 5.0, 20.0])).unsqueeze(0)
    atom_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    delta = adapter(z_emb, frac, lattice, atom_mask)

    assert delta.shape == frac.shape
    assert torch.allclose(delta[0, 1], torch.zeros(3), atol=1e-6)
