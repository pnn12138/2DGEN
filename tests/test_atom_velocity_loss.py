from __future__ import annotations

from types import SimpleNamespace

import torch

from twodgen.common.atom_diffusion import AtomDiffusionConfig, AtomVelocityLoss
from twodgen.common.crystal import lattice_to_gram6


class DummyModel(torch.nn.Module):
    def __init__(self, base_frac: torch.Tensor, base_gram6: torch.Tensor) -> None:
        super().__init__()
        self.mask_id = 0
        self.cfg = SimpleNamespace(g_scale=1.0)
        self._base_frac = base_frac
        self._base_gram6 = base_gram6

    def forward(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        cell: torch.Tensor,
        atom_mask: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        counts_vector: torch.Tensor | None = None,
        return_geom: bool = False,
        **kwargs,
    ):
        bsz, n, _ = frac.shape
        num_elements = max(int(z.max().item()), 1)
        logits_z = torch.zeros(
            (bsz, n, num_elements + 1),
            device=frac.device,
            dtype=frac.dtype,
        )
        pred_f = self._base_frac.expand_as(frac)
        pred_g = self._base_gram6.expand_as(cell)
        if return_geom:
            geom_preds = {}
            if "uv_angle" in kwargs and kwargs["uv_angle"] is not None:
                geom_preds["uv_angle"] = kwargs["uv_angle"]
            if "z_norm" in kwargs and kwargs["z_norm"] is not None:
                geom_preds["z_norm"] = kwargs["z_norm"]
            if "lattice_param" in kwargs and kwargs["lattice_param"] is not None:
                geom_preds["lattice_param"] = kwargs["lattice_param"]
            if "slab_t" in kwargs and kwargs["slab_t"] is not None:
                geom_preds["t"] = kwargs["slab_t"]
            return pred_f, pred_g, logits_z, geom_preds
        return pred_f, pred_g, logits_z


def test_atom_velocity_loss_cross_vacuum_and_vacuum_loss() -> None:
    torch.manual_seed(0)
    lattice = torch.diag(torch.tensor([5.0, 5.0, 10.0])).unsqueeze(0)
    gram6 = lattice_to_gram6(lattice)
    frac = torch.tensor([[[0.1, 0.1, 0.1], [0.1, 0.1, 0.9]]], dtype=torch.float32)
    z = torch.tensor([[1, 1]], dtype=torch.long)
    atom_mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    cfg = AtomDiffusionConfig(
        lambda_vacuum=1.0,
        vacuum_min=15.0,
        vacuum_loss_power=1,
        lambda_cross_vacuum=1.0,
        cross_vacuum_bond_cut=9.0,
        cross_vacuum_power=1,
        use_uncertainty_weighting=False,
    )
    loss_fn = AtomVelocityLoss(cfg, mask_token_id=0)
    model = DummyModel(frac, gram6)

    loss, _, _, _, metrics = loss_fn(model, z, frac, atom_mask, gram6)

    assert torch.isfinite(loss)
    assert float(metrics["loss_vacuum"]) > 0.0
    assert float(metrics["loss_cross_vacuum"]) > 0.0
    assert float(metrics["cross_vacuum_rate"]) > 0.0
