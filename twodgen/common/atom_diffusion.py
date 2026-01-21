from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from twodgen.common.crystal import gram6_to_cholesky6
from twodgen.common.crystal import cholesky6_to_gram6, gram6_to_lattice, frac_mic_dist

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
    noise_scale_zn: float = 0.3
    noise_scale_lat: float = 1.0
    noise_scale_t: float = 1.0
    lambda_comp: float = 1.0
    comp_loss_mode: str = "l1"  # l1 | cosine
    lambda_vacuum: float = 0.0
    vacuum_min: float = 15.0
    vacuum_loss_power: int = 2
    lambda_angle: float = 0.1
    angle_min: float = 30.0
    angle_max: float = 150.0
    lambda_cond: float = 0.01
    cond_max: float = 1e3
    lambda_chol_bound: float = 0.0
    chol_bound_margin: float = 0.2
    chol_bound_power: int = 2
    lambda_expand_collision: float = 0.0
    expand_min_dist_cut: float = 1.5
    lambda_volume: float = 0.0
    volume_min: float = 1.0
    lambda_c_len: float = 0.0
    c_len_min: float = 15.0
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
            self.s_comp = nn.Parameter(torch.zeros(()))
            self.s_vacuum = nn.Parameter(torch.zeros(()))
            self.s_angle = nn.Parameter(torch.zeros(()))
            self.s_cond = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _counts_from_z(
        z: torch.Tensor,
        mask: torch.Tensor,
        num_elements: int,
    ) -> torch.Tensor:
        counts = torch.zeros(z.shape[0], num_elements, device=z.device, dtype=torch.float32)
        valid = (mask > 0.5) & (z > 0) & (z <= num_elements)
        if not valid.any():
            return counts
        z_valid = z[valid].long()
        batch_idx = torch.arange(z.shape[0], device=z.device).unsqueeze(1).expand_as(valid)[valid]
        elem_idx = z_valid - 1
        counts.index_put_(
            (batch_idx, elem_idx),
            torch.ones_like(elem_idx, dtype=counts.dtype),
            accumulate=True,
        )
        return counts

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
            pred_x0_f, pred_x0_g, logits_z, geom_preds = model(
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
            pred_x0_f, pred_x0_g, logits_z = model(
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

        if self.cfg.mode == "flow":
            denom_f = t_expand_f.clamp_min(self.cfg.t_eps)
            denom_g = t_expand_g.clamp_min(self.cfg.t_eps)
            pred_v_f = (frac_t - pred_x0_f) / denom_f
            pred_v_g = (cell_t - pred_x0_g) / denom_g
        else:
            denom_f = (1.0 - t_expand_f).clamp_min(self.cfg.t_eps)
            denom_g = (1.0 - t_expand_g).clamp_min(self.cfg.t_eps)
            pred_v_f = (pred_x0_f - frac_t) / denom_f
            pred_v_g = (pred_x0_g - cell_t) / denom_g

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
                uv_pred_x0 = geom_preds["uv_angle"]
                if self.cfg.mode == "flow":
                    denom_uv = t_expand_uv.clamp_min(self.cfg.t_eps)
                    uv_pred = (uv_angle_t - uv_pred_x0) / denom_uv
                else:
                    denom_uv = (1.0 - t_expand_uv).clamp_min(self.cfg.t_eps)
                    uv_pred = (uv_pred_x0 - uv_angle_t) / denom_uv
                loss_uv = (uv_pred - v_uv) ** 2
                loss_uv = (loss_uv * atom_mask_f).sum() / atom_mask_f.sum().clamp_min(1.0)
            if z_norm is not None and v_zn is not None:
                zn_pred_x0 = geom_preds["z_norm"]
                if self.cfg.mode == "flow":
                    denom_zn = t_expand_zn.clamp_min(self.cfg.t_eps)
                    zn_pred = (z_norm_t - zn_pred_x0) / denom_zn
                else:
                    denom_zn = (1.0 - t_expand_zn).clamp_min(self.cfg.t_eps)
                    zn_pred = (zn_pred_x0 - z_norm_t) / denom_zn
                loss_zn = (zn_pred - v_zn) ** 2
                loss_zn = (loss_zn * atom_mask).sum() / atom_mask.sum().clamp_min(1.0)
            if lattice_param is not None and v_lat is not None:
                lat_pred_x0 = geom_preds["lattice_param"]
                if self.cfg.mode == "flow":
                    denom_lat = t_expand_lat.clamp_min(self.cfg.t_eps)
                    lat_pred = (lattice_param_t - lat_pred_x0) / denom_lat
                else:
                    denom_lat = (1.0 - t_expand_lat).clamp_min(self.cfg.t_eps)
                    lat_pred = (lat_pred_x0 - lattice_param_t) / denom_lat
                loss_lat = F.mse_loss(lat_pred, v_lat)
            if slab_t is not None and v_t is not None:
                t_pred_x0 = geom_preds["t"]
                if self.cfg.mode == "flow":
                    denom_t = t_expand_t.clamp_min(self.cfg.t_eps)
                    if denom_t.ndim > 1:
                        denom_t = denom_t.squeeze(-1)
                    t_pred = (slab_t_t - t_pred_x0) / denom_t
                else:
                    denom_t = (1.0 - t_expand_t).clamp_min(self.cfg.t_eps)
                    if denom_t.ndim > 1:
                        denom_t = denom_t.squeeze(-1)
                    t_pred = (t_pred_x0 - slab_t_t) / denom_t
                loss_t = F.mse_loss(t_pred, v_t)

        if masked_pos.any():
            loss_z = F.cross_entropy(logits_z[masked_pos], z[masked_pos], reduction="mean")
        else:
            loss_z = torch.tensor(0.0, device=device)

        loss_comp = torch.tensor(0.0, device=device)
        if (counts_vector is not None) or masked_pos.any():
            num_elements = int(logits_z.shape[-1] - 1)
            if counts_vector is None:
                target_counts = self._counts_from_z(z, atom_mask, num_elements)
            else:
                target = counts_vector.float()
                if target.ndim == 1:
                    target = target.unsqueeze(0)
                if target.shape[-1] < num_elements:
                    raise ValueError(
                        f"counts_vector dim {target.shape[-1]} < num_elements {num_elements}"
                    )
                target_counts = target[..., :num_elements]

            atom_valid = atom_mask > 0.5
            unmasked = atom_valid & (~masked_pos)
            known_counts = self._counts_from_z(z, unmasked.float(), num_elements)
            remaining = (target_counts - known_counts).clamp_min(0.0)

            if masked_pos.any():
                probs = torch.softmax(logits_z, dim=-1)[..., 1 : num_elements + 1]
                expected = (probs * masked_pos.float().unsqueeze(-1)).sum(dim=1)
                denom = remaining.sum(dim=-1).clamp_min(1.0)
                mode = str(self.cfg.comp_loss_mode).lower()
                if mode == "cosine":
                    exp_norm = expected / expected.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    tgt_norm = remaining / remaining.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    loss_comp = (1.0 - (exp_norm * tgt_norm).sum(dim=-1)).mean()
                else:
                    loss_comp = (expected - remaining).abs().sum(dim=-1).div(denom).mean()

        loss_vacuum = torch.tensor(0.0, device=device)
        if self.cfg.lambda_vacuum > 0.0 and self.cfg.vacuum_min > 0.0:
            g_scale = getattr(getattr(model, "cfg", None), "g_scale", 1.0)
            if self.cfg.cell_rep == "cholesky6":
                gram6_pred = cholesky6_to_gram6(pred_x0_g, log_min=self.cfg.chol_log_min, log_max=self.cfg.chol_log_max)
            else:
                gram6_pred = pred_x0_g
            lattice = gram6_to_lattice(gram6_pred * float(g_scale))
            lengths = torch.linalg.norm(lattice, dim=-1)
            c_idx = torch.argmax(lengths, dim=-1)
            c_len = lengths.gather(1, c_idx.unsqueeze(1)).squeeze(1)
            frac_wrap = frac - torch.floor(frac)
            idx = c_idx.view(-1, 1, 1).expand(-1, frac_wrap.shape[1], 1)
            frac_c = torch.take_along_dim(frac_wrap, idx, dim=2).squeeze(2)
            vacuums = torch.empty((bsz,), device=device, dtype=frac.dtype)
            for b in range(bsz):
                vals = frac_c[b][atom_mask[b] > 0.5]
                if vals.numel() == 0 or not torch.isfinite(c_len[b]):
                    vacuums[b] = torch.tensor(float("nan"), device=device, dtype=frac.dtype)
                    continue
                vals, _ = torch.sort(vals)
                if vals.numel() == 1:
                    max_gap = torch.tensor(1.0, device=device, dtype=frac.dtype)
                else:
                    gaps = vals[1:] - vals[:-1]
                    wrap_gap = 1.0 - (vals[-1] - vals[0])
                    max_gap = torch.max(torch.max(gaps), wrap_gap)
                vacuums[b] = max_gap * c_len[b]
            vac_min = float(self.cfg.vacuum_min)
            delta = (vac_min - vacuums).clamp_min(0.0)
            power = int(self.cfg.vacuum_loss_power)
            if power <= 1:
                loss_vacuum = torch.nanmean(delta / max(vac_min, 1e-8))
            else:
                loss_vacuum = torch.nanmean((delta / max(vac_min, 1e-8)) ** power)

        loss_angle = torch.tensor(0.0, device=device)
        loss_cond = torch.tensor(0.0, device=device)
        loss_chol_bound = torch.tensor(0.0, device=device)
        loss_expand_collision = torch.tensor(0.0, device=device)
        loss_volume = torch.tensor(0.0, device=device)
        loss_c_len = torch.tensor(0.0, device=device)
        angle_out_rate = torch.tensor(float("nan"), device=device)
        cond_mean = torch.tensor(float("nan"), device=device)
        chol_bound_rate = torch.tensor(float("nan"), device=device)
        min_dist_pred_mean = torch.tensor(float("nan"), device=device)
        min_dist_pred_p10 = torch.tensor(float("nan"), device=device)
        if self.cfg.lambda_volume > 0.0 or self.cfg.lambda_c_len > 0.0:
            g_scale = getattr(getattr(model, "cfg", None), "g_scale", 1.0)
            if self.cfg.cell_rep == "cholesky6":
                gram6_pred_vol = cholesky6_to_gram6(
                    pred_x0_g, log_min=self.cfg.chol_log_min, log_max=self.cfg.chol_log_max
                )
            else:
                gram6_pred_vol = pred_x0_g
            lattice_vol = gram6_to_lattice(gram6_pred_vol * float(g_scale))
            lengths_vol = torch.linalg.norm(lattice_vol, dim=-1)
            c_len_val = lengths_vol.max(dim=-1).values
            if self.cfg.lambda_c_len > 0.0:
                c_min = float(self.cfg.c_len_min)
                delta_len = (c_min - c_len_val).clamp_min(0.0)
                denom = max(c_min, 1e-8)
                loss_c_len = torch.nanmean((delta_len / denom) ** 2)
            if self.cfg.lambda_volume > 0.0:
                volume = torch.abs(torch.linalg.det(lattice_vol))
                vol_min = float(self.cfg.volume_min)
                delta_vol = (vol_min - volume).clamp_min(0.0)
                denom = max(vol_min, 1e-8)
                loss_volume = torch.nanmean((delta_vol / denom) ** 2)
        if self.cfg.lambda_angle > 0.0 or self.cfg.lambda_cond > 0.0:
            g_scale = getattr(getattr(model, "cfg", None), "g_scale", 1.0)
            if self.cfg.cell_rep == "cholesky6":
                gram6_pred = cholesky6_to_gram6(
                    pred_x0_g, log_min=self.cfg.chol_log_min, log_max=self.cfg.chol_log_max
                )
            else:
                gram6_pred = pred_x0_g
            lattice = gram6_to_lattice(gram6_pred * float(g_scale))
            lengths = torch.linalg.norm(lattice, dim=-1)
            valid = torch.isfinite(lengths).all(dim=-1) & (lengths > 0).all(dim=-1)
            if valid.any():
                a_vec = lattice[valid, 0]
                b_vec = lattice[valid, 1]
                c_vec = lattice[valid, 2]
                cos_alpha = (b_vec * c_vec).sum(dim=-1) / (
                    b_vec.norm(dim=-1) * c_vec.norm(dim=-1)
                ).clamp_min(1e-8)
                cos_beta = (a_vec * c_vec).sum(dim=-1) / (
                    a_vec.norm(dim=-1) * c_vec.norm(dim=-1)
                ).clamp_min(1e-8)
                cos_gamma = (a_vec * b_vec).sum(dim=-1) / (
                    a_vec.norm(dim=-1) * b_vec.norm(dim=-1)
                ).clamp_min(1e-8)
                cos_alpha = cos_alpha.clamp(-1.0, 1.0)
                cos_beta = cos_beta.clamp(-1.0, 1.0)
                cos_gamma = cos_gamma.clamp(-1.0, 1.0)
                alpha = torch.rad2deg(torch.acos(cos_alpha))
                beta = torch.rad2deg(torch.acos(cos_beta))
                gamma = torch.rad2deg(torch.acos(cos_gamma))
                angle_min = float(self.cfg.angle_min)
                angle_max = float(self.cfg.angle_max)
                viol = torch.stack(
                    [
                        (angle_min - alpha).clamp_min(0.0),
                        (alpha - angle_max).clamp_min(0.0),
                        (angle_min - beta).clamp_min(0.0),
                        (beta - angle_max).clamp_min(0.0),
                        (angle_min - gamma).clamp_min(0.0),
                        (gamma - angle_max).clamp_min(0.0),
                    ],
                    dim=-1,
                )
                if self.cfg.lambda_angle > 0.0:
                    loss_angle = (viol ** 2).mean()
                angle_out_rate = (viol.max(dim=-1).values > 0).float().mean()

                if self.cfg.lambda_cond > 0.0:
                    gram = lattice[valid] @ lattice[valid].transpose(-1, -2)
                    eigvals = torch.linalg.eigvalsh(gram)
                    cond = eigvals.max(dim=-1).values / eigvals.min(dim=-1).values.clamp_min(1e-8)
                    cond_mean = cond.mean()
                    cond_max = float(self.cfg.cond_max)
                    cond_violation = (cond - cond_max).clamp_min(0.0) / max(cond_max, 1e-8)
                    loss_cond = (cond_violation ** 2).mean()

        if self.cfg.lambda_chol_bound > 0.0:
            chol_min = self.cfg.chol_log_min
            chol_max = self.cfg.chol_log_max
            if chol_min is not None or chol_max is not None:
                if self.cfg.cell_rep == "cholesky6":
                    chol_params = pred_x0_g
                else:
                    chol_params = gram6_to_cholesky6(pred_x0_g, log_min=None, log_max=None)
                diag = chol_params[:, :3]
                margin = max(float(self.cfg.chol_bound_margin), 0.0)
                lower = torch.zeros_like(diag)
                upper = torch.zeros_like(diag)
                if chol_min is not None:
                    lower = (float(chol_min) + margin - diag).clamp_min(0.0)
                if chol_max is not None:
                    upper = (diag - (float(chol_max) - margin)).clamp_min(0.0)
                violation = lower + upper
                denom = max(margin, 1e-6)
                if int(self.cfg.chol_bound_power) <= 1:
                    loss_chol_bound = (violation / denom).mean()
                else:
                    loss_chol_bound = ((violation / denom) ** int(self.cfg.chol_bound_power)).mean()
                chol_bound_rate = (violation > 0).any(dim=1).float().mean()

        if self.cfg.lambda_expand_collision > 0.0 and self.cfg.expand_min_dist_cut > 0.0:
            g_scale = getattr(getattr(model, "cfg", None), "g_scale", 1.0)
            if self.cfg.cell_rep == "cholesky6":
                gram6_pred = cholesky6_to_gram6(
                    pred_x0_g, log_min=self.cfg.chol_log_min, log_max=self.cfg.chol_log_max
                )
            else:
                gram6_pred = pred_x0_g
            lattice = gram6_to_lattice(gram6_pred * float(g_scale))
            frac_pred = pred_x0_f - torch.floor(pred_x0_f)
            pbc_mask = getattr(getattr(model, "cfg", None), "pbc_mask", (1, 1, 1))
            dist = frac_mic_dist(frac_pred, lattice, atom_mask, pbc_mask=pbc_mask)
            min_dist = dist.amin(dim=(1, 2))
            cut = float(self.cfg.expand_min_dist_cut)
            delta = (cut - min_dist).clamp_min(0.0) / max(cut, 1e-8)
            severity = (cut - min_dist).clamp_min(0.0)
            severity_norm = severity / max(cut, 1e-8)
            loss_expand_collision = torch.nanmean((delta * (1.0 + severity_norm)) ** 2)
            valid = torch.isfinite(min_dist)
            if valid.any():
                min_dist_pred_mean = min_dist[valid].mean()
                min_dist_pred_p10 = torch.quantile(min_dist[valid], 0.1)

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
            if loss_comp is not None and self.cfg.lambda_comp > 0.0:
                loss = loss + torch.exp(-self.s_comp) * loss_comp + self.s_comp
            if self.cfg.lambda_vacuum > 0.0:
                loss = loss + torch.exp(-self.s_vacuum) * loss_vacuum + self.s_vacuum
            if self.cfg.lambda_angle > 0.0:
                loss = loss + torch.exp(-self.s_angle) * loss_angle + self.s_angle
            if self.cfg.lambda_cond > 0.0:
                loss = loss + torch.exp(-self.s_cond) * loss_cond + self.s_cond
            if self.cfg.lambda_volume > 0.0:
                loss = loss + self.cfg.lambda_volume * loss_volume
            if self.cfg.lambda_c_len > 0.0:
                loss = loss + self.cfg.lambda_c_len * loss_c_len
            if self.cfg.lambda_chol_bound > 0.0:
                loss = loss + self.cfg.lambda_chol_bound * loss_chol_bound
            if self.cfg.lambda_expand_collision > 0.0:
                loss = loss + self.cfg.lambda_expand_collision * loss_expand_collision
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
            if self.cfg.lambda_comp > 0.0:
                loss = loss + self.cfg.lambda_comp * loss_comp
            if self.cfg.lambda_vacuum > 0.0:
                loss = loss + self.cfg.lambda_vacuum * loss_vacuum
            if self.cfg.lambda_angle > 0.0:
                loss = loss + self.cfg.lambda_angle * loss_angle
            if self.cfg.lambda_cond > 0.0:
                loss = loss + self.cfg.lambda_cond * loss_cond
            if self.cfg.lambda_volume > 0.0:
                loss = loss + self.cfg.lambda_volume * loss_volume
            if self.cfg.lambda_c_len > 0.0:
                loss = loss + self.cfg.lambda_c_len * loss_c_len
            if self.cfg.lambda_chol_bound > 0.0:
                loss = loss + self.cfg.lambda_chol_bound * loss_chol_bound
            if self.cfg.lambda_expand_collision > 0.0:
                loss = loss + self.cfg.lambda_expand_collision * loss_expand_collision
        metrics = {
            "loss_f": loss_f.detach(),
            "loss_g": loss_g.detach(),
            "loss_z": loss_z.detach(),
            "loss_uv": loss_uv.detach(),
            "loss_zn": loss_zn.detach(),
            "loss_lat": loss_lat.detach(),
            "loss_t": loss_t.detach(),
            "loss_comp": loss_comp.detach(),
            "loss_vacuum": loss_vacuum.detach(),
            "loss_angle": loss_angle.detach(),
            "loss_cond": loss_cond.detach(),
            "loss_chol_bound": loss_chol_bound.detach(),
            "loss_expand_collision": loss_expand_collision.detach(),
            "loss_volume": loss_volume.detach(),
            "loss_c_len": loss_c_len.detach(),
            "pred_angle_out_rate": angle_out_rate.detach(),
            "pred_cond_mean": cond_mean.detach(),
            "chol_bound_rate": chol_bound_rate.detach(),
            "min_dist_pred_mean": min_dist_pred_mean.detach(),
            "min_dist_pred_p10": min_dist_pred_p10.detach(),
            "pred_x0_f_mean": pred_x0_f.detach().mean(),
            "pred_x0_f_std": pred_x0_f.detach().std(),
            "pred_v_f_mean": pred_v_f.detach().mean(),
            "pred_v_f_std": pred_v_f.detach().std(),
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
                    "s_comp": self.s_comp.detach(),
                    "s_vacuum": self.s_vacuum.detach(),
                    "s_angle": self.s_angle.detach(),
                    "s_cond": self.s_cond.detach(),
                }
            )
        return loss, pred_v_f, pred_v_g, logits_z, metrics


__all__ = ["AtomDiffusionConfig", "AtomVelocityLoss", "logit_normal_sample", "expand_t", "mask_schedule"]
