from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from twodgen.common.crystal import frac_mic_dist_with_shifts


class EgNNTailAdapter(nn.Module):
    def __init__(
        self,
        z_embed_dim: int,
        hidden_dim: int,
        pbc_mask: Tuple[int, int, int],
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(z_embed_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(z_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.pbc_mask = pbc_mask

    def forward(
        self,
        z_emb: torch.Tensor,
        frac: torch.Tensor,
        lattice: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n, _ = frac.shape
        if n == 0:
            return torch.zeros_like(frac)
        dist, shifts = frac_mic_dist_with_shifts(
            frac, lattice, atom_mask, pbc_mask=self.pbc_mask
        )
        df = frac[:, :, None, :] - frac[:, None, :, :] - shifts.to(frac.dtype)
        dr = torch.einsum("bijm,bmn->bijn", df, lattice)
        dist = torch.linalg.norm(dr, dim=-1).clamp_min(1e-6)

        h_i = z_emb.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = z_emb.unsqueeze(1).expand(-1, n, -1, -1)
        edge_in = torch.cat([h_i, h_j, dist.unsqueeze(-1)], dim=-1)
        weights = self.edge_mlp(edge_in).squeeze(-1)

        valid = atom_mask > 0.5
        pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        weights = weights.masked_fill(~pair_mask, 0.0)
        weights = weights - torch.diag_embed(torch.diagonal(weights, dim1=1, dim2=2))

        delta_cart = torch.sum(weights.unsqueeze(-1) * dr, dim=2)
        node_gate = self.node_mlp(z_emb).squeeze(-1)
        delta_cart = delta_cart * node_gate.unsqueeze(-1)
        try:
            inv_lattice = torch.linalg.inv(lattice)
        except RuntimeError:
            return torch.zeros_like(frac)
        delta_frac = torch.einsum("bij,bjk->bik", delta_cart, inv_lattice)
        delta_frac = delta_frac * atom_mask.unsqueeze(-1)
        return delta_frac * self.scale
