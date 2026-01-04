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
    lambda_uv: float = 1.0
    lambda_zn: float = 1.0
    lambda_lat: float = 0.5
    lambda_t: float = 0.5
    noise_scale_uv: float = 1.0
    noise_scale_zn: float = 1.0
    noise_scale_lat: float = 1.0
    noise_scale_t: float = 1.0
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
            self.s_uv = nn.Parameter(torch.zeros(()))
            self.s_zn = nn.Parameter(torch.zeros(()))
            self.s_lat = nn.Parameter(torch.zeros(()))
            self.s_t = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        model: nn.Module,
        z: torch.Tensor,
        frac: torch.Tensor,
        atom_mask: torch.Tensor,
        gram6: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        counts_vector: Optional[torch.Tensor] = None,
        uv_angle: Optional[torch.Tensor] = None,
        z_norm: Optional[torch.Tensor] = None,
        lattice_param: Optional[torch.Tensor] = None,
        slab_t: Optional[torch.Tensor] = None,
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
        noise_uv = None
        uv_angle_t = None
        v_uv = None
        if uv_angle is not None:
            noise_uv = torch.randn_like(uv_angle) * self.cfg.noise_scale_uv
            t_expand_uv = expand_t(t, uv_angle.ndim)
            if self.cfg.mode == "flow":
                uv_angle_t = t_expand_uv * noise_uv + (1.0 - t_expand_uv) * uv_angle
                v_uv = noise_uv - uv_angle
            else:
                uv_angle_t = t_expand_uv * uv_angle + (1.0 - t_expand_uv) * noise_uv
                denom_uv = (1.0 - t_expand_uv).clamp_min(self.cfg.t_eps)
                v_uv = (uv_angle - uv_angle_t) / denom_uv

        noise_zn = None
        z_norm_t = None
        v_zn = None
        if z_norm is not None:
            z_norm_in = z_norm.unsqueeze(-1)
            noise_zn = torch.randn_like(z_norm_in) * self.cfg.noise_scale_zn
            t_expand_zn = expand_t(t, z_norm_in.ndim)
            if self.cfg.mode == "flow":
                z_norm_t = t_expand_zn * noise_zn + (1.0 - t_expand_zn) * z_norm_in
                v_zn = noise_zn - z_norm_in
            else:
                z_norm_t = t_expand_zn * z_norm_in + (1.0 - t_expand_zn) * noise_zn
                denom_zn = (1.0 - t_expand_zn).clamp_min(self.cfg.t_eps)
                v_zn = (z_norm_in - z_norm_t) / denom_zn
            z_norm_t = z_norm_t.squeeze(-1)
            v_zn = v_zn.squeeze(-1)
        if self.cfg.cell_rep == "cholesky6":
            cell = gram6_to_cholesky6(
                gram6, log_min=self.cfg.chol_log_min, log_max=self.cfg.chol_log_max
            )
        else:
            cell = gram6
        noise_g = torch.randn_like(cell) * self.cfg.noise_scale_g
        noise_lat = None
        lattice_param_t = None
        v_lat = None
        if lattice_param is not None:
            noise_lat = torch.randn_like(lattice_param) * self.cfg.noise_scale_lat
            t_expand_lat = expand_t(t, lattice_param.ndim)
            if self.cfg.mode == "flow":
                lattice_param_t = t_expand_lat * noise_lat + (1.0 - t_expand_lat) * lattice_param
                v_lat = noise_lat - lattice_param
            else:
                lattice_param_t = t_expand_lat * lattice_param + (1.0 - t_expand_lat) * noise_lat
                denom_lat = (1.0 - t_expand_lat).clamp_min(self.cfg.t_eps)
                v_lat = (lattice_param - lattice_param_t) / denom_lat
        noise_t = None
        slab_t_t = None
        v_t = None
        if slab_t is not None:
            if slab_t.ndim == 1:
                slab_t_in = slab_t.unsqueeze(-1)
            else:
                slab_t_in = slab_t
            noise_t = torch.randn_like(slab_t_in) * self.cfg.noise_scale_t
            t_expand_t = expand_t(t, slab_t_in.ndim)
            if self.cfg.mode == "flow":
                slab_t_t = t_expand_t * noise_t + (1.0 - t_expand_t) * slab_t_in
                v_t = noise_t - slab_t_in
            else:
                slab_t_t = t_expand_t * slab_t_in + (1.0 - t_expand_t) * noise_t
                denom_t = (1.0 - t_expand_t).clamp_min(self.cfg.t_eps)
                v_t = (slab_t_in - slab_t_t) / denom_t
            slab_t_t = slab_t_t.squeeze(-1)
            v_t = v_t.squeeze(-1)
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
        counts_in = counts_vector
        if (cond is not None or counts_vector is not None) and self.cfg.cond_drop_prob > 0.0:
            drop = torch.rand(bsz, device=device) < self.cfg.cond_drop_prob
            if drop.any():
                if cond is not None:
                    cond_in = cond.clone()
                    cond_in[drop] = 0.0
                if counts_vector is not None:
                    counts_in = counts_vector.clone()
                    counts_in[drop] = 0.0
        geom_preds = None
        if uv_angle is not None or z_norm is not None or lattice_param is not None or slab_t is not None:
            pred_v_f, pred_v_g, logits_z, geom_preds = model(
                z_masked,
                frac_t,
                cell_t,
                atom_mask,
                t,
                cond_in,
                counts_in,
                uv_angle=uv_angle_t,
                z_norm=z_norm_t,
                lattice_param=lattice_param_t,
                slab_t=slab_t_t,
                return_geom=True,
                nbr_idx=nbr_idx,
                nbr_mask=nbr_mask,
                dist_nbr=dist_nbr,
            )
        else:
            pred_v_f, pred_v_g, logits_z = model(
                z_masked,
                frac_t,
                cell_t,
                atom_mask,
                t,
                cond_in,
                counts_in,
                nbr_idx=nbr_idx,
                nbr_mask=nbr_mask,
                dist_nbr=dist_nbr,
            )

        atom_mask_f = atom_mask.unsqueeze(-1)
        loss_f = (pred_v_f - v_f) ** 2
        loss_f = (loss_f * atom_mask_f).sum() / atom_mask_f.sum().clamp_min(1.0)

        loss_g = F.mse_loss(pred_v_g, v_g)
        loss_uv = torch.tensor(0.0, device=device)
        loss_zn = torch.tensor(0.0, device=device)
        loss_lat = torch.tensor(0.0, device=device)
        loss_t = torch.tensor(0.0, device=device)
        if geom_preds is not None:
            if uv_angle is not None and v_uv is not None:
                uv_pred = geom_preds["uv_angle"]
                loss_uv = (uv_pred - v_uv) ** 2
                loss_uv = (loss_uv * atom_mask_f).sum() / atom_mask_f.sum().clamp_min(1.0)
            if z_norm is not None and v_zn is not None:
                zn_pred = geom_preds["z_norm"]
                loss_zn = (zn_pred - v_zn) ** 2
                loss_zn = (loss_zn * atom_mask).sum() / atom_mask.sum().clamp_min(1.0)
            if lattice_param is not None and v_lat is not None:
                lat_pred = geom_preds["lattice_param"]
                loss_lat = F.mse_loss(lat_pred, v_lat)
            if slab_t is not None and v_t is not None:
                t_pred = geom_preds["t"]
                loss_t = F.mse_loss(t_pred, v_t)

        if masked_pos.any():
            loss_z = F.cross_entropy(logits_z[masked_pos], z[masked_pos], reduction="mean")
        else:
            loss_z = torch.tensor(0.0, device=device)

        if self.cfg.use_uncertainty_weighting:
            loss = torch.exp(-self.s_f) * loss_f + self.s_f + torch.exp(-self.s_g) * loss_g + self.s_g
            if masked_pos.any():
                loss = loss + torch.exp(-self.s_z) * loss_z + self.s_z
            if uv_angle is not None:
                loss = loss + torch.exp(-self.s_uv) * loss_uv + self.s_uv
            if z_norm is not None:
                loss = loss + torch.exp(-self.s_zn) * loss_zn + self.s_zn
            if lattice_param is not None:
                loss = loss + torch.exp(-self.s_lat) * loss_lat + self.s_lat
            if slab_t is not None:
                loss = loss + torch.exp(-self.s_t) * loss_t + self.s_t
        else:
            loss = loss_f + self.cfg.lambda_z * loss_z + self.cfg.lambda_g * loss_g
            if uv_angle is not None:
                loss = loss + self.cfg.lambda_uv * loss_uv
            if z_norm is not None:
                loss = loss + self.cfg.lambda_zn * loss_zn
            if lattice_param is not None:
                loss = loss + self.cfg.lambda_lat * loss_lat
            if slab_t is not None:
                loss = loss + self.cfg.lambda_t * loss_t
        metrics = {
            "loss_f": loss_f.detach(),
            "loss_g": loss_g.detach(),
            "loss_z": loss_z.detach(),
            "loss_uv": loss_uv.detach(),
            "loss_zn": loss_zn.detach(),
            "loss_lat": loss_lat.detach(),
            "loss_t": loss_t.detach(),
        }
        if self.cfg.use_uncertainty_weighting:
            metrics.update(
                {
                    "s_f": self.s_f.detach(),
                    "s_g": self.s_g.detach(),
                    "s_z": self.s_z.detach(),
                    "s_uv": self.s_uv.detach(),
                    "s_zn": self.s_zn.detach(),
                    "s_lat": self.s_lat.detach(),
                    "s_t": self.s_t.detach(),
                }
            )
        return loss, pred_v_f, pred_v_g, logits_z, metrics


__all__ = ["AtomDiffusionConfig", "AtomVelocityLoss", "logit_normal_sample", "expand_t", "mask_schedule"]
