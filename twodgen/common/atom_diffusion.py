from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from twodgen.common.crystal import gram6_to_cholesky6

@dataclass
class AtomDiffusionConfig:
    P_mean: float = -1.2
    P_std: float = 1.2
    t_eps: float = 1e-5
    noise_scale_f: float = 1.0
    noise_scale_g: float = 1.0
    p_mask_min: float = 0.05
    p_mask_max: float = 0.6
    lambda_z: float = 1.0
    lambda_g: float = 0.5
    use_uncertainty_weighting: bool = True
    mode: str = "diffusion"  # diffusion | flow
    cell_rep: str = "gram6"  # gram6 | cholesky6
    chol_log_min: Optional[float] = None
    chol_log_max: Optional[float] = None
    cell_init: str = "gaussian"  # gaussian | iso
    cell_init_scale: Optional[float] = None
    cell_init_noise: Optional[float] = None
    cond_drop_prob: float = 0.0


def logit_normal_sample(batch_size: int, device: torch.device, P_mean: float, P_std: float) -> torch.Tensor:
    z = torch.randn(batch_size, device=device) * P_std + P_mean
    return torch.sigmoid(z)


def expand_t(t: torch.Tensor, ndims: int) -> torch.Tensor:
    return t.view(-1, *([1] * (ndims - 1)))


def mask_schedule(t: torch.Tensor, p_min: float, p_max: float, mode: str) -> torch.Tensor:
    if mode == "flow":
        return p_min + (p_max - p_min) * t
    return p_max - (p_max - p_min) * t


class AtomVelocityLoss(nn.Module):
    def __init__(self, cfg: AtomDiffusionConfig, mask_token_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.mask_token_id = mask_token_id
        if cfg.use_uncertainty_weighting:
            self.s_f = nn.Parameter(torch.zeros(()))
            self.s_g = nn.Parameter(torch.zeros(()))
            self.s_z = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        model: nn.Module,
        z: torch.Tensor,
        frac: torch.Tensor,
        atom_mask: torch.Tensor,
        gram6: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        nbr_idx: Optional[torch.Tensor] = None,
        nbr_mask: Optional[torch.Tensor] = None,
        dist_nbr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        device = frac.device
        bsz = frac.shape[0]

        if self.cfg.mode == "flow":
            t = torch.rand(bsz, device=device)
        else:
            t = logit_normal_sample(bsz, device, self.cfg.P_mean, self.cfg.P_std)
        t_expand_f = expand_t(t, frac.ndim)
        t_expand_g = expand_t(t, gram6.ndim)

        noise_f = torch.randn_like(frac) * self.cfg.noise_scale_f
        if self.cfg.cell_rep == "cholesky6":
            cell = gram6_to_cholesky6(
                gram6, log_min=self.cfg.chol_log_min, log_max=self.cfg.chol_log_max
            )
        else:
            cell = gram6
        noise_g = torch.randn_like(cell) * self.cfg.noise_scale_g
        if self.cfg.mode == "flow":
            frac_t = t_expand_f * noise_f + (1.0 - t_expand_f) * frac
            cell_t = t_expand_g * noise_g + (1.0 - t_expand_g) * cell
            v_f = noise_f - frac
            v_g = noise_g - cell
        else:
            frac_t = t_expand_f * frac + (1.0 - t_expand_f) * noise_f
            cell_t = t_expand_g * cell + (1.0 - t_expand_g) * noise_g
            denom_f = (1.0 - t_expand_f).clamp_min(self.cfg.t_eps)
            denom_g = (1.0 - t_expand_g).clamp_min(self.cfg.t_eps)
            v_f = (frac - frac_t) / denom_f
            v_g = (cell - cell_t) / denom_g

        p_mask = mask_schedule(t, self.cfg.p_mask_min, self.cfg.p_mask_max, self.cfg.mode)
        rand = torch.rand_like(z.float())
        masked_pos = (rand < p_mask.unsqueeze(1)) & (atom_mask > 0.5)
        z_masked = z.clone()
        z_masked[masked_pos] = self.mask_token_id

        cond_in = cond
        if cond is not None and self.cfg.cond_drop_prob > 0.0:
            drop = torch.rand(bsz, device=device) < self.cfg.cond_drop_prob
            if drop.any():
                cond_in = cond.clone()
                cond_in[drop] = 0.0
        pred_v_f, pred_v_g, logits_z = model(
            z_masked,
            frac_t,
            cell_t,
            atom_mask,
            t,
            cond_in,
            nbr_idx=nbr_idx,
            nbr_mask=nbr_mask,
            dist_nbr=dist_nbr,
        )

        atom_mask_f = atom_mask.unsqueeze(-1)
        loss_f = (pred_v_f - v_f) ** 2
        loss_f = (loss_f * atom_mask_f).sum() / atom_mask_f.sum().clamp_min(1.0)

        loss_g = F.mse_loss(pred_v_g, v_g)

        if masked_pos.any():
            loss_z = F.cross_entropy(logits_z[masked_pos], z[masked_pos], reduction="mean")
        else:
            loss_z = torch.tensor(0.0, device=device)

        if self.cfg.use_uncertainty_weighting:
            loss = torch.exp(-self.s_f) * loss_f + self.s_f + torch.exp(-self.s_g) * loss_g + self.s_g
            if masked_pos.any():
                loss = loss + torch.exp(-self.s_z) * loss_z + self.s_z
        else:
            loss = loss_f + self.cfg.lambda_z * loss_z + self.cfg.lambda_g * loss_g
        metrics = {
            "loss_f": loss_f.detach(),
            "loss_g": loss_g.detach(),
            "loss_z": loss_z.detach(),
        }
        if self.cfg.use_uncertainty_weighting:
            metrics.update(
                {
                    "s_f": self.s_f.detach(),
                    "s_g": self.s_g.detach(),
                    "s_z": self.s_z.detach(),
                }
            )
        return loss, pred_v_f, pred_v_g, logits_z, metrics


__all__ = ["AtomDiffusionConfig", "AtomVelocityLoss", "logit_normal_sample", "expand_t", "mask_schedule"]
