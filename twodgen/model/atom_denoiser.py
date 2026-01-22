from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import nn

from twodgen.common.atom_diffusion import AtomDiffusionConfig, AtomVelocityLoss, expand_t, mask_schedule
from twodgen.common.crystal import (
    cholesky6_to_gram6,
    cholesky6_to_lattice,
    clip_lattice,
    frac_mic_dist,
    frac_mic_dist_with_shifts,
    gram6_to_cholesky6,
    gram6_to_lattice,
    lattice_to_gram6,
    niggli_reduce_lattice,
    reduce_lattice_simple,
)
from twodgen.model.atom_transformer import AtomTransformer, AtomTransformerConfig


@dataclass
class AtomDenoiserConfig:
    model: AtomTransformerConfig = field(default_factory=AtomTransformerConfig)
    diffusion: AtomDiffusionConfig = field(default_factory=AtomDiffusionConfig)
    sampling_method: str = "euler"
    num_sampling_steps: int = 20
    neighbor_update_steps: int = 1
    v_min: float = 1e-3
    v_max: float = 1e3
    cond_max: float = 1e3
    reduce_lattice: bool = False
    niggli_reduce: bool = False
    project_each_step: bool = False
    project_geometry: bool = False
    z_norm_clip: float = 1.5
    min_dist_iter: int = 0
    min_dist_strength: float = 0.03
    min_dist_cut: float = 1.5
    min_dist_train_cut: float = 1.5
    min_dist_train_weight: float = 0.02
    chol_log_relax: float = 0.0


class AtomDenoiser(nn.Module):
    def __init__(self, cfg: AtomDenoiserConfig = AtomDenoiserConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = AtomTransformer(cfg.model)
        self.loss_fn = AtomVelocityLoss(cfg.diffusion, self.model.mask_id)

    def forward(
        self,
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
        loss, pred_v_f, pred_v_g, logits_z, metrics = self.loss_fn(
            self.model,
            z,
            frac,
            atom_mask,
            gram6,
            cond,
            counts_vector,
            uv_angle,
            z_norm,
            lattice_param,
            slab_t,
            nbr_idx,
            nbr_mask,
            dist_nbr,
        )
        if self.training and self.cfg.min_dist_train_weight > 0.0:
            lattice = gram6_to_lattice(gram6 * self.cfg.model.g_scale)
            dist = frac_mic_dist(frac, lattice, atom_mask, pbc_mask=self.cfg.model.pbc_mask)
            cut = float(self.cfg.min_dist_train_cut)
            delta = (cut - dist).clamp_min(0.0)
            # Collision-prioritized reduction: penalize only the worst (closest) pair per structure.
            # This makes "one bad short bond" visible even when many pairs are already fine.
            penalty_per = (delta ** 2).amax(dim=(1, 2))
            penalty = penalty_per.mean()
            loss = loss + self.cfg.min_dist_train_weight * penalty
            metrics["loss_min_dist"] = penalty.detach()
        return loss, pred_v_f, pred_v_g, logits_z, metrics

    def _relaxed_chol_bounds(
        self,
    ) -> tuple[Optional[float | tuple[float, float, float]], Optional[float | tuple[float, float, float]]]:
        min_val = self.cfg.model.chol_log_min_vec if self.cfg.model.chol_log_min_vec is not None else self.cfg.model.chol_log_min
        max_val = self.cfg.model.chol_log_max_vec if self.cfg.model.chol_log_max_vec is not None else self.cfg.model.chol_log_max
        relax = float(self.cfg.chol_log_relax)
        if relax <= 0.0:
            return min_val, max_val
        if min_val is not None:
            if isinstance(min_val, (tuple, list)):
                min_val = tuple(float(v) - relax for v in min_val)  # type: ignore[assignment]
            else:
                min_val = float(min_val) - relax
        if max_val is not None:
            if isinstance(max_val, (tuple, list)):
                max_val = tuple(float(v) + relax for v in max_val)  # type: ignore[assignment]
            else:
                max_val = float(max_val) + relax
        return min_val, max_val

    def _predict_velocity(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        atom_mask: torch.Tensor,
        gram6: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        counts_vector: Optional[torch.Tensor] = None,
        uv_angle: Optional[torch.Tensor] = None,
        z_norm: Optional[torch.Tensor] = None,
        lattice_param: Optional[torch.Tensor] = None,
        slab_t: Optional[torch.Tensor] = None,
        return_geom: bool = False,
        step: Optional[int] = None,
        cache_every: Optional[int] = None,
        nbr_idx: Optional[torch.Tensor] = None,
        nbr_mask: Optional[torch.Tensor] = None,
        dist_nbr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        outputs = self.model(
            z,
            frac,
            gram6,
            atom_mask,
            t,
            cond,
            counts_vector,
            uv_angle=uv_angle,
            z_norm=z_norm,
            lattice_param=lattice_param,
            slab_t=slab_t,
            return_geom=return_geom,
            step=step,
            cache_every=cache_every,
            nbr_idx=nbr_idx,
            nbr_mask=nbr_mask,
            dist_nbr=dist_nbr,
        )
        if return_geom:
            pred_x0_f, pred_x0_g, logits_z, geom_preds = outputs  # type: ignore[misc]
        else:
            pred_x0_f, pred_x0_g, logits_z = outputs  # type: ignore[misc]

        if self.cfg.diffusion.mode == "flow":
            denom_f = expand_t(t, frac.ndim).clamp_min(self.cfg.diffusion.t_eps)
            denom_g = expand_t(t, gram6.ndim).clamp_min(self.cfg.diffusion.t_eps)
            pred_v_f = (frac - pred_x0_f) / denom_f
            pred_v_g = (gram6 - pred_x0_g) / denom_g
        else:
            denom_f = expand_t(1.0 - t, frac.ndim).clamp_min(self.cfg.diffusion.t_eps)
            denom_g = expand_t(1.0 - t, gram6.ndim).clamp_min(self.cfg.diffusion.t_eps)
            pred_v_f = (pred_x0_f - frac) / denom_f
            pred_v_g = (pred_x0_g - gram6) / denom_g

        if return_geom:
            if uv_angle is not None:
                denom_uv = expand_t(t, uv_angle.ndim if uv_angle.ndim > 0 else 2).clamp_min(self.cfg.diffusion.t_eps)
                if self.cfg.diffusion.mode == "flow":
                    geom_preds["uv_angle"] = (uv_angle - geom_preds["uv_angle"]) / denom_uv
                else:
                    denom_uv = expand_t(1.0 - t, uv_angle.ndim).clamp_min(self.cfg.diffusion.t_eps)
                    geom_preds["uv_angle"] = (geom_preds["uv_angle"] - uv_angle) / denom_uv
            if z_norm is not None:
                denom_zn = expand_t(t, z_norm.ndim).clamp_min(self.cfg.diffusion.t_eps)
                if self.cfg.diffusion.mode == "flow":
                    geom_preds["z_norm"] = (z_norm - geom_preds["z_norm"]) / denom_zn
                else:
                    denom_zn = expand_t(1.0 - t, z_norm.ndim).clamp_min(self.cfg.diffusion.t_eps)
                    geom_preds["z_norm"] = (geom_preds["z_norm"] - z_norm) / denom_zn
            if lattice_param is not None:
                denom_lat = expand_t(t, lattice_param.ndim).clamp_min(self.cfg.diffusion.t_eps)
                if self.cfg.diffusion.mode == "flow":
                    geom_preds["lattice_param"] = (lattice_param - geom_preds["lattice_param"]) / denom_lat
                else:
                    denom_lat = expand_t(1.0 - t, lattice_param.ndim).clamp_min(self.cfg.diffusion.t_eps)
                    geom_preds["lattice_param"] = (geom_preds["lattice_param"] - lattice_param) / denom_lat
            if slab_t is not None:
                denom_t = expand_t(t, slab_t.ndim).clamp_min(self.cfg.diffusion.t_eps)
                if self.cfg.diffusion.mode == "flow":
                    geom_preds["t"] = (slab_t - geom_preds["t"]) / denom_t
                else:
                    denom_t = expand_t(1.0 - t, slab_t.ndim).clamp_min(self.cfg.diffusion.t_eps)
                    geom_preds["t"] = (geom_preds["t"] - slab_t) / denom_t
            return pred_v_f, pred_v_g, logits_z, geom_preds
        return pred_v_f, pred_v_g, logits_z

    @torch.no_grad()
    def _euler_step(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        atom_mask: torch.Tensor,
        gram6: torch.Tensor,
        uv_angle: Optional[torch.Tensor],
        z_norm: Optional[torch.Tensor],
        lattice_param: Optional[torch.Tensor],
        slab_t: Optional[torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        counts_vector: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        cache_every: Optional[int] = None,
        return_geom: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        outputs = self._predict_velocity(
            z,
            frac,
            atom_mask,
            gram6,
            t,
            cond,
            counts_vector,
            uv_angle=uv_angle,
            z_norm=z_norm,
            lattice_param=lattice_param,
            slab_t=slab_t,
            return_geom=return_geom,
            step=step,
            cache_every=cache_every,
        )
        if return_geom:
            pred_v_f, pred_v_g, logits_z, geom_preds = outputs  # type: ignore[misc]
        else:
            pred_v_f, pred_v_g, logits_z = outputs  # type: ignore[misc]
        delta = expand_t(t_next - t, frac.ndim)
        frac = frac + delta * pred_v_f
        gram6 = gram6 + expand_t(t_next - t, gram6.ndim) * pred_v_g
        if return_geom and uv_angle is not None and z_norm is not None:
            uv_angle = uv_angle + delta * geom_preds["uv_angle"]
            delta_zn = expand_t(t_next - t, z_norm.ndim)
            z_norm = z_norm + delta_zn * geom_preds["z_norm"]
        if return_geom and lattice_param is not None:
            delta_lat = expand_t(t_next - t, lattice_param.ndim)
            lattice_param = lattice_param + delta_lat * geom_preds["lattice_param"]
        if return_geom and slab_t is not None:
            delta_t = expand_t(t_next - t, slab_t.ndim)
            slab_t = slab_t + delta_t * geom_preds["t"]
        return frac, gram6, logits_z, uv_angle, z_norm, lattice_param, slab_t

    def _project_geometry_step(
        self,
        uv_angle: torch.Tensor,
        z_norm: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        uv1 = uv_angle[..., :2]
        uv2 = uv_angle[..., 2:]
        eps = 1e-8
        uv1 = uv1 / uv1.norm(dim=-1, keepdim=True).clamp_min(eps)
        uv2 = uv2 / uv2.norm(dim=-1, keepdim=True).clamp_min(eps)
        uv_angle = torch.cat([uv1, uv2], dim=-1)
        z_norm = torch.clamp(z_norm, min=-self.cfg.z_norm_clip, max=self.cfg.z_norm_clip)
        mask = atom_mask.unsqueeze(-1)
        uv_angle = uv_angle * mask
        z_norm = z_norm * atom_mask
        return uv_angle, z_norm

    def _project_step(
        self, frac: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frac = frac - torch.floor(frac)
        if self.cfg.diffusion.cell_rep == "cholesky6":
            chol_min, chol_max = self._relaxed_chol_bounds()
            if chol_min is not None or chol_max is not None:
                cell = cell.clone()
                min_bound = (
                    torch.tensor(chol_min, device=cell.device, dtype=cell.dtype)
                    if isinstance(chol_min, (tuple, list))
                    else chol_min
                )
                max_bound = (
                    torch.tensor(chol_max, device=cell.device, dtype=cell.dtype)
                    if isinstance(chol_max, (tuple, list))
                    else chol_max
                )
                cell[:, :3] = torch.clamp(cell[:, :3], min=min_bound, max=max_bound)
            lattice = cholesky6_to_lattice(
                cell, log_min=chol_min, log_max=chol_max
            )
            lattice = clip_lattice(lattice, self.cfg.v_min, self.cfg.v_max, self.cfg.cond_max)
            gram6 = lattice_to_gram6(lattice)
            cell = gram6_to_cholesky6(gram6, log_min=chol_min, log_max=chol_max)
            return frac, cell
        lattice = gram6_to_lattice(cell * self.cfg.model.g_scale)
        lattice = clip_lattice(lattice, self.cfg.v_min, self.cfg.v_max, self.cfg.cond_max)
        cell = lattice_to_gram6(lattice) / self.cfg.model.g_scale
        return frac, cell

    @torch.no_grad()
    def _apply_min_dist_repulsion(
        self,
        frac: torch.Tensor,
        cell: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.min_dist_iter <= 0 or self.cfg.min_dist_strength <= 0.0:
            return frac
        if self.cfg.min_dist_cut <= 0.0:
            return frac
        if self.cfg.diffusion.cell_rep == "cholesky6":
            log_min = self.cfg.model.chol_log_min_vec if self.cfg.model.chol_log_min_vec is not None else self.cfg.model.chol_log_min
            log_max = self.cfg.model.chol_log_max_vec if self.cfg.model.chol_log_max_vec is not None else self.cfg.model.chol_log_max
            gram6 = cholesky6_to_gram6(
                cell, log_min=log_min, log_max=log_max
            )
        else:
            gram6 = cell
        lattice = gram6_to_lattice(gram6 * self.cfg.model.g_scale)
        try:
            inv_lattice = torch.linalg.inv(lattice)
        except RuntimeError:
            return frac

        cut = float(self.cfg.min_dist_cut)
        strength = float(self.cfg.min_dist_strength)
        for _ in range(int(self.cfg.min_dist_iter)):
            dist, shifts = frac_mic_dist_with_shifts(
                frac, lattice, atom_mask, pbc_mask=self.cfg.model.pbc_mask
            )
            mask = dist < cut
            if not torch.any(mask):
                break
            df = frac[:, :, None, :] - frac[:, None, :, :]
            mic_df = df - shifts.to(df.dtype)
            dr = torch.einsum("bijm,bmn->bijn", mic_df, lattice)
            dist_safe = dist.clamp_min(1e-8)
            direction = dr / dist_safe.unsqueeze(-1)
            push = (cut - dist_safe).clamp_min(0.0) / max(cut, 1e-8)
            push = torch.where(mask, push, torch.zeros_like(push))
            push = push * strength
            delta_cart = direction * push.unsqueeze(-1)
            disp_cart = delta_cart.sum(dim=2)
            disp_frac = torch.einsum("bim,bmn->bin", disp_cart, inv_lattice)
            mask = atom_mask.unsqueeze(-1)
            disp_frac = disp_frac * mask
            denom = mask.sum(dim=1).clamp_min(1.0)
            mean_disp = disp_frac.sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
            disp_frac = disp_frac - mean_disp
            frac = frac + disp_frac
            frac = frac - torch.floor(frac)
        return frac

    @torch.no_grad()
    def _heun_step(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        atom_mask: torch.Tensor,
        gram6: torch.Tensor,
        uv_angle: Optional[torch.Tensor],
        z_norm: Optional[torch.Tensor],
        lattice_param: Optional[torch.Tensor],
        slab_t: Optional[torch.Tensor],
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        counts_vector: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        cache_every: Optional[int] = None,
        return_geom: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        outputs = self._predict_velocity(
            z,
            frac,
            atom_mask,
            gram6,
            t,
            cond,
            counts_vector,
            uv_angle=uv_angle,
            z_norm=z_norm,
            lattice_param=lattice_param,
            slab_t=slab_t,
            return_geom=return_geom,
            step=step,
            cache_every=cache_every,
        )
        if return_geom:
            pred_v_f, pred_v_g, logits_z, geom_preds = outputs  # type: ignore[misc]
        else:
            pred_v_f, pred_v_g, logits_z = outputs  # type: ignore[misc]
        delta = expand_t(t_next - t, frac.ndim)
        frac_euler = frac + delta * pred_v_f
        gram_euler = gram6 + expand_t(t_next - t, gram6.ndim) * pred_v_g
        uv_euler = None
        zn_euler = None
        lat_euler = None
        t_euler = None
        if return_geom and uv_angle is not None and z_norm is not None:
            uv_euler = uv_angle + delta * geom_preds["uv_angle"]
            delta_zn = expand_t(t_next - t, z_norm.ndim)
            zn_euler = z_norm + delta_zn * geom_preds["z_norm"]
        if return_geom and lattice_param is not None:
            delta_lat = expand_t(t_next - t, lattice_param.ndim)
            lat_euler = lattice_param + delta_lat * geom_preds["lattice_param"]
        if return_geom and slab_t is not None:
            delta_t = expand_t(t_next - t, slab_t.ndim)
            t_euler = slab_t + delta_t * geom_preds["t"]
        outputs_next = self._predict_velocity(
            z,
            frac_euler,
            atom_mask,
            gram_euler,
            t_next,
            cond,
            uv_angle=uv_euler if return_geom else None,
            z_norm=zn_euler if return_geom else None,
            lattice_param=lat_euler if return_geom else None,
            slab_t=t_euler if return_geom else None,
            return_geom=return_geom,
            step=step,
            cache_every=cache_every,
        )
        if return_geom:
            pred_v_f_next, pred_v_g_next, _, geom_preds_next = outputs_next  # type: ignore[misc]
            pred_uv = 0.5 * (geom_preds["uv_angle"] + geom_preds_next["uv_angle"])
            pred_zn = 0.5 * (geom_preds["z_norm"] + geom_preds_next["z_norm"])
            pred_lat = 0.5 * (geom_preds["lattice_param"] + geom_preds_next["lattice_param"])
            pred_t = 0.5 * (geom_preds["t"] + geom_preds_next["t"])
        else:
            pred_v_f_next, pred_v_g_next, _ = outputs_next  # type: ignore[misc]
        pred_v_f = 0.5 * (pred_v_f + pred_v_f_next)
        pred_v_g = 0.5 * (pred_v_g + pred_v_g_next)
        frac = frac + delta * pred_v_f
        gram6 = gram6 + expand_t(t_next - t, gram6.ndim) * pred_v_g
        if return_geom and uv_angle is not None and z_norm is not None:
            uv_angle = uv_angle + delta * pred_uv
            delta_zn = expand_t(t_next - t, z_norm.ndim)
            z_norm = z_norm + delta_zn * pred_zn
        if return_geom and lattice_param is not None:
            delta_lat = expand_t(t_next - t, lattice_param.ndim)
            lattice_param = lattice_param + delta_lat * pred_lat
        if return_geom and slab_t is not None:
            delta_t = expand_t(t_next - t, slab_t.ndim)
            slab_t = slab_t + delta_t * pred_t
        return frac, gram6, logits_z, uv_angle, z_norm, lattice_param, slab_t

    @torch.no_grad()
    def generate(
        self,
        num_atoms: int,
        max_atoms: int,
        batch_size: int,
        steps: Optional[int] = None,
        method: Optional[str] = None,
        z_sampling: str = "argmax",
        z_temperature: float = 1.0,
        z_top_k: int = 10,
        z_top_p: float = 0.9,
        cond: Optional[torch.Tensor] = None,
        counts_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        method = method or self.cfg.sampling_method
        steps = steps if steps is not None else self.cfg.num_sampling_steps
        device = self.model.z_embed.weight.device

        atom_mask = torch.zeros(batch_size, max_atoms, device=device)
        atom_mask[:, :num_atoms] = 1.0

        z = torch.zeros(batch_size, max_atoms, dtype=torch.long, device=device)
        z[:, :num_atoms] = self.model.mask_id

        frac = torch.randn(batch_size, max_atoms, 3, device=device) * self.cfg.diffusion.noise_scale_f
        frac = frac * atom_mask.unsqueeze(-1)
        uv_angle = None
        z_norm = None
        lattice_param = None
        slab_t = None
        if self.cfg.project_geometry:
            uv_angle = torch.randn(batch_size, max_atoms, 4, device=device) * self.cfg.diffusion.noise_scale_uv
            z_norm = torch.randn(batch_size, max_atoms, device=device) * self.cfg.diffusion.noise_scale_zn
            uv_angle = uv_angle * atom_mask.unsqueeze(-1)
            z_norm = z_norm * atom_mask
            lattice_param = torch.randn(batch_size, 3, device=device) * self.cfg.diffusion.noise_scale_lat
            slab_t = torch.randn(batch_size, device=device) * self.cfg.diffusion.noise_scale_t
        if self.cfg.diffusion.cell_rep == "cholesky6" and self.cfg.diffusion.cell_init == "iso":
            scale = 1.0 if self.cfg.diffusion.cell_init_scale is None else self.cfg.diffusion.cell_init_scale
            log_s = torch.log(torch.tensor(scale, device=device))
            chol_min = self.cfg.model.chol_log_min_vec if self.cfg.model.chol_log_min_vec is not None else self.cfg.model.chol_log_min
            chol_max = self.cfg.model.chol_log_max_vec if self.cfg.model.chol_log_max_vec is not None else self.cfg.model.chol_log_max
            if chol_min is not None or chol_max is not None:
                min_scalar = min(chol_min) if isinstance(chol_min, (tuple, list)) else chol_min
                max_scalar = max(chol_max) if isinstance(chol_max, (tuple, list)) else chol_max
                log_s = torch.clamp(log_s, min=min_scalar, max=max_scalar)
            base = torch.zeros(batch_size, 6, device=device)
            base[:, :3] = log_s
            noise_scale = 0.2 if self.cfg.diffusion.cell_init_noise is None else self.cfg.diffusion.cell_init_noise
            noise = torch.randn(batch_size, 6, device=device) * noise_scale
            cell = base + noise
        else:
            cell = torch.randn(batch_size, 6, device=device) * self.cfg.diffusion.noise_scale_g

        if self.cfg.diffusion.mode == "flow":
            t_schedule = torch.linspace(1.0, 0.0, steps + 1, device=device)
        else:
            t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)

        remaining_counts = None
        if counts_vector is not None:
            counts = counts_vector.to(device=device)
            if counts.ndim == 1:
                counts = counts.unsqueeze(0)
            if counts.shape[0] != batch_size:
                raise ValueError(
                    f"counts_vector batch {counts.shape[0]} does not match batch_size={batch_size}"
                )
            num_elements = int(self.cfg.model.num_elements)
            if counts.shape[-1] < num_elements:
                raise ValueError(
                    f"counts_vector dim {counts.shape[-1]} < num_elements {num_elements}"
                )
            counts = counts[..., :num_elements].long().clamp_min(0)
            if torch.any(counts.sum(dim=-1) != int(num_atoms)):
                raise ValueError("counts_vector sum must match num_atoms when provided.")
            remaining_counts = counts.clone()

        def sample_z(logits: torch.Tensor) -> torch.Tensor:
            logits = logits.clone()
            logits[..., 0] = float("-inf")
            if z_sampling == "argmax":
                return torch.argmax(logits, dim=-1)
            if z_sampling == "temperature":
                temp = max(z_temperature, 1e-4)
                probs = torch.softmax(logits / temp, dim=-1)
                return torch.multinomial(probs, num_samples=1).squeeze(-1)
            if z_sampling == "topk":
                k = max(1, min(z_top_k, logits.shape[-1]))
                vals, idx = torch.topk(logits, k=k, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                choice = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            if z_sampling == "topp":
                p = min(max(z_top_p, 0.0), 1.0)
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs <= p
                mask[..., 0] = True
                neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
                filtered = torch.where(mask, sorted_logits, neg_inf)
                probs = torch.softmax(filtered, dim=-1)
                choice = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return sorted_idx.gather(-1, choice.unsqueeze(-1)).squeeze(-1)
            raise ValueError(f"Unknown z sampling method: {z_sampling}")

        def sample_z_constrained(logits_1d: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
            allowed = remaining > 0
            if not torch.any(allowed):
                return torch.zeros((), dtype=torch.long, device=logits_1d.device)
            logits = logits_1d.clone()
            logits[0] = float("-inf")
            logits[1:] = torch.where(allowed, logits[1:], torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype))
            if not torch.isfinite(logits).any():
                pick = torch.nonzero(allowed, as_tuple=False).flatten()[0]
                return pick + 1
            return sample_z(logits.unsqueeze(0)).squeeze(0)

        for i in range(steps):
            t = t_schedule[i].expand(batch_size)
            t_next = t_schedule[i + 1].expand(batch_size)
            if method == "euler":
                frac, cell, logits_z, uv_angle, z_norm, lattice_param, slab_t = self._euler_step(
                    z,
                    frac,
                    atom_mask,
                    cell,
                    uv_angle,
                    z_norm,
                    lattice_param,
                    slab_t,
                    t,
                    t_next,
                    cond,
                    counts_vector,
                    step=i,
                    cache_every=self.cfg.neighbor_update_steps,
                    return_geom=self.cfg.project_geometry,
                )
            elif method == "heun":
                frac, cell, logits_z, uv_angle, z_norm, lattice_param, slab_t = self._heun_step(
                    z,
                    frac,
                    atom_mask,
                    cell,
                    uv_angle,
                    z_norm,
                    lattice_param,
                    slab_t,
                    t,
                    t_next,
                    cond,
                    counts_vector,
                    step=i,
                    cache_every=self.cfg.neighbor_update_steps,
                    return_geom=self.cfg.project_geometry,
                )
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            if self.cfg.project_each_step:
                frac, cell = self._project_step(frac, cell)
            if self.cfg.project_geometry and uv_angle is not None and z_norm is not None:
                uv_angle, z_norm = self._project_geometry_step(uv_angle, z_norm, atom_mask)

            p_mask = mask_schedule(
                t_next, self.cfg.diffusion.p_mask_min, self.cfg.diffusion.p_mask_max, self.cfg.diffusion.mode
            )
            if i == steps - 1:
                p_mask = torch.zeros_like(p_mask)
            target_mask = torch.round(p_mask * num_atoms).long().clamp(min=0, max=num_atoms)
            for b in range(batch_size):
                masked_idx = (z[b, :num_atoms] == self.model.mask_id).nonzero(as_tuple=False).flatten()
                keep = target_mask[b].item()
                if masked_idx.numel() <= keep:
                    continue
                logits = logits_z[b, :num_atoms]
                probs = torch.softmax(logits, dim=-1).max(dim=-1).values
                scores = probs[masked_idx]
                _, order = torch.topk(scores, k=masked_idx.numel() - keep, largest=True)
                reveal_idx = masked_idx[order]
                if remaining_counts is None:
                    z_pred = sample_z(logits[reveal_idx])
                    z[b, reveal_idx] = z_pred
                else:
                    for idx in reveal_idx.tolist():
                        token = sample_z_constrained(logits[idx], remaining_counts[b])
                        if token.item() > 0:
                            remaining_counts[b, token.item() - 1] -= 1
                        z[b, idx] = token

        frac = frac - torch.floor(frac)
        frac_pre = frac.clone()
        if self.cfg.diffusion.cell_rep == "cholesky6":
            chol_min, chol_max = self._relaxed_chol_bounds()
            gram6 = cholesky6_to_gram6(
                cell, log_min=chol_min, log_max=chol_max
            )
        else:
            gram6 = cell
        lattice = gram6_to_lattice(gram6 * self.cfg.model.g_scale)
        dist_pre = frac_mic_dist(frac_pre, lattice, atom_mask, pbc_mask=self.cfg.model.pbc_mask)
        min_dist_pre = dist_pre.amin(dim=(1, 2))

        frac = self._apply_min_dist_repulsion(frac, cell, atom_mask)
        dist_post = frac_mic_dist(frac, lattice, atom_mask, pbc_mask=self.cfg.model.pbc_mask)
        min_dist_post = dist_post.amin(dim=(1, 2))
        if self.cfg.diffusion.cell_rep == "cholesky6":
            chol_min, chol_max = self._relaxed_chol_bounds()
            gram6 = cholesky6_to_gram6(
                cell, log_min=chol_min, log_max=chol_max
            )
        else:
            gram6 = cell
        return z, frac, gram6, atom_mask, lattice_param, slab_t, min_dist_pre, min_dist_post

    @torch.no_grad()
    def gram6_to_lattice(self, gram6: torch.Tensor) -> torch.Tensor:
        lattice = gram6_to_lattice(gram6 * self.cfg.model.g_scale)
        if self.cfg.reduce_lattice:
            lattice = reduce_lattice_simple(lattice)
        if self.cfg.niggli_reduce:
            lattice = niggli_reduce_lattice(lattice)
        return lattice


__all__ = ["AtomDenoiserConfig", "AtomDenoiser"]
