from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch
from torch import nn
import numpy as np

from twodgen.common.atom_diffusion import AtomDiffusionConfig, AtomVelocityLoss, expand_t, mask_schedule
from twodgen.common.crystal import (
    cholesky6_to_gram6,
    cholesky6_to_lattice,
    frac_mic_dist,
    frac_mic_dist_with_shifts,
    gram6_to_cholesky6,
    gram6_to_lattice,
    lattice_to_gram6,
    niggli_reduce_lattice,
    reduce_lattice_simple,
)
from twodgen.common.geometry_torch import choose_vacuum_axis_torch
from twodgen.model.atom_transformer import AtomTransformer, AtomTransformerConfig
from twodgen.model.cell_net import CellNet
from twodgen.model.tail_adapters import EgNNTailAdapter


@dataclass
class AtomDenoiserConfig:
    model: AtomTransformerConfig = field(default_factory=AtomTransformerConfig)
    diffusion: AtomDiffusionConfig = field(default_factory=AtomDiffusionConfig)
    sampling_method: str = "euler"
    num_sampling_steps: int = 20
    neighbor_update_steps: int = 1
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
    z_clamp: bool = False
    z_clamp_min_t: float = 5.0
    z_clamp_max_t: Optional[float] = None
    symmetry_loss_weight: float = 0.0
    symmetry_symprec: float = 1e-2


class AtomDenoiser(nn.Module):
    def __init__(self, cfg: AtomDenoiserConfig = AtomDenoiserConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = AtomTransformer(cfg.model)
        self.loss_fn = AtomVelocityLoss(cfg.diffusion, self.model.mask_id)
        self.tail_adapter: Optional[nn.Module] = None
        self.cell_net: Optional[nn.Module] = None
        if cfg.model.tail_adapter and cfg.model.tail_adapter != "none":
            if cfg.model.tail_adapter == "egnn":
                self.tail_adapter = EgNNTailAdapter(
                    cfg.model.z_embed_dim,
                    cfg.model.tail_hidden_dim,
                    cfg.model.pbc_mask,
                    init_scale=cfg.model.tail_scale,
                )
            else:
                raise ValueError(f"Unknown tail_adapter={cfg.model.tail_adapter!r}")
        if cfg.diffusion.cell_init == "cellnet":
            if cfg.model.cond_dim <= 0:
                raise ValueError("cellnet requires cond_dim > 0.")
            self.cell_net = CellNet(
                cfg.model.cond_dim, cfg.diffusion.cell_net_hidden_dim, out_dim=6
            )

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
        spacegroup_number: Optional[torch.Tensor] = None,
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
            min_dist_train_weight=self.cfg.min_dist_train_weight if self.training else 0.0,
            min_dist_train_cut=self.cfg.min_dist_train_cut,
        )
        if (
            self.cfg.symmetry_loss_weight > 0.0
            and spacegroup_number is not None
        ):
            sym_loss, sym_rate = self._symmetry_residual_loss(
                frac, gram6, atom_mask, z, spacegroup_number
            )
            metrics["loss_symmetry"] = sym_loss.detach()
            metrics["symmetry_violation_rate"] = sym_rate.detach()
        return loss, pred_v_f, pred_v_g, logits_z, metrics

    def _symmetry_residual_loss(
        self,
        frac: torch.Tensor,
        cell: torch.Tensor,
        atom_mask: torch.Tensor,
        z: torch.Tensor,
        spacegroup_number: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            import spglib  # type: ignore
        except Exception:
            zero = torch.tensor(0.0, device=frac.device, dtype=frac.dtype)
            return zero, zero
        lattice = self._cell_to_lattice(cell).detach().cpu().numpy()
        frac_np = frac.detach().cpu().numpy()
        mask_np = atom_mask.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        target = spacegroup_number.detach().cpu().numpy().reshape(-1)
        violations = []
        for i in range(frac_np.shape[0]):
            if target[i] <= 0:
                continue
            mask_i = (mask_np[i] > 0.5) & (z_np[i] > 0)
            if not np.any(mask_i):
                continue
            cell_tuple = (lattice[i], frac_np[i][mask_i], z_np[i][mask_i].astype(int))
            try:
                dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=float(self.cfg.symmetry_symprec))
            except Exception:
                dataset = None
            if dataset is None:
                violations.append(1.0)
                continue
            sg_number = int(dataset.get("number"))
            violations.append(0.0 if sg_number == int(target[i]) else 1.0)
        if not violations:
            zero = torch.tensor(0.0, device=frac.device, dtype=frac.dtype)
            return zero, zero
        mean_violation = float(np.mean(violations))
        loss = torch.tensor(mean_violation, device=frac.device, dtype=frac.dtype)
        rate = torch.tensor(mean_violation, device=frac.device, dtype=frac.dtype)
        return loss, rate

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

        if self.tail_adapter is not None:
            lattice = self._cell_to_lattice(gram6)
            z_emb = self.model.z_embed(z)
            delta = self.tail_adapter(z_emb, pred_x0_f, lattice, atom_mask)
            pred_x0_f = pred_x0_f + delta

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

    def _cell_to_lattice(self, cell: torch.Tensor) -> torch.Tensor:
        if self.cfg.diffusion.cell_rep == "cholesky6":
            chol_min = self.cfg.model.chol_log_min_vec if self.cfg.model.chol_log_min_vec is not None else self.cfg.model.chol_log_min
            chol_max = self.cfg.model.chol_log_max_vec if self.cfg.model.chol_log_max_vec is not None else self.cfg.model.chol_log_max
            lattice = cholesky6_to_lattice(cell, log_min=chol_min, log_max=chol_max)
            lattice = lattice * float(self.cfg.model.g_scale) ** 0.5
            return lattice
        return gram6_to_lattice(cell * self.cfg.model.g_scale)

    def _predict_velocity_guided(
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
        guidance_scale: float = 1.0,
    ):
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
        if guidance_scale <= 1.0 or (cond is None and counts_vector is None):
            return outputs
        outputs_u = self._predict_velocity(
            z,
            frac,
            atom_mask,
            gram6,
            t,
            None,
            None,
            uv_angle=uv_angle,
            z_norm=z_norm,
            lattice_param=lattice_param,
            slab_t=slab_t,
            return_geom=return_geom,
            step=step,
            cache_every=cache_every,
        )

        def _blend(cond_out: torch.Tensor, uncond_out: torch.Tensor) -> torch.Tensor:
            return uncond_out + float(guidance_scale) * (cond_out - uncond_out)

        if return_geom:
            pred_v_f, pred_v_g, logits_z, geom_preds = outputs  # type: ignore[misc]
            pred_v_f_u, pred_v_g_u, logits_z_u, geom_preds_u = outputs_u  # type: ignore[misc]
            pred_v_f = _blend(pred_v_f, pred_v_f_u)
            pred_v_g = _blend(pred_v_g, pred_v_g_u)
            logits_z = _blend(logits_z, logits_z_u)
            guided_geom = {k: _blend(geom_preds[k], geom_preds_u[k]) for k in geom_preds}
            return pred_v_f, pred_v_g, logits_z, guided_geom

        pred_v_f, pred_v_g, logits_z = outputs  # type: ignore[misc]
        pred_v_f_u, pred_v_g_u, logits_z_u = outputs_u  # type: ignore[misc]
        pred_v_f = _blend(pred_v_f, pred_v_f_u)
        pred_v_g = _blend(pred_v_g, pred_v_g_u)
        logits_z = _blend(logits_z, logits_z_u)
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
        guidance_scale: float = 1.0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        outputs = self._predict_velocity_guided(
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
            guidance_scale=guidance_scale,
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
            lattice = cholesky6_to_lattice(cell, log_min=chol_min, log_max=chol_max)
            sqrt_g = float(self.cfg.model.g_scale) ** 0.5
            lattice_phys = lattice * sqrt_g
            gram6 = lattice_to_gram6(lattice_phys) / float(self.cfg.model.g_scale)
            cell = gram6_to_cholesky6(gram6, log_min=chol_min, log_max=chol_max)
            return frac, cell
        lattice = gram6_to_lattice(cell * self.cfg.model.g_scale)
        cell = lattice_to_gram6(lattice) / self.cfg.model.g_scale
        return frac, cell

    def _z_clamp_step(
        self,
        frac: torch.Tensor,
        cell: torch.Tensor,
        atom_mask: torch.Tensor,
        slab_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.cfg.z_clamp or slab_t is None:
            return frac
        if self.cfg.diffusion.cell_rep == "cholesky6":
            lattice = cholesky6_to_lattice(cell)
            lattice = lattice * float(self.cfg.model.g_scale) ** 0.5
        else:
            lattice = gram6_to_lattice(cell * self.cfg.model.g_scale)
        c_idx, c_len, _ = choose_vacuum_axis_torch(lattice)
        c_len = c_len.clamp_min(1e-6)

        t_val = slab_t
        if self.cfg.z_clamp_max_t is not None:
            t_val = torch.minimum(t_val, torch.tensor(self.cfg.z_clamp_max_t, device=t_val.device))
        t_val = torch.clamp(t_val, min=self.cfg.z_clamp_min_t)
        t_frac = (t_val / c_len).clamp(max=1.0)
        lower = 0.5 - 0.5 * t_frac
        upper = 0.5 + 0.5 * t_frac

        for b in range(frac.shape[0]):
            idx = int(c_idx[b].item())
            lo = lower[b].item()
            hi = upper[b].item()
            frac[b, :, idx] = torch.clamp(frac[b, :, idx], min=lo, max=hi)
        frac = frac * atom_mask.unsqueeze(-1)
        return frac

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
        guidance_scale: float = 1.0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        outputs = self._predict_velocity_guided(
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
            guidance_scale=guidance_scale,
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
        outputs_next = self._predict_velocity_guided(
            z,
            frac_euler,
            atom_mask,
            gram_euler,
            t_next,
            cond,
            counts_vector,
            uv_angle=uv_euler if return_geom else None,
            z_norm=zn_euler if return_geom else None,
            lattice_param=lat_euler if return_geom else None,
            slab_t=t_euler if return_geom else None,
            return_geom=return_geom,
            step=step,
            cache_every=cache_every,
            guidance_scale=guidance_scale,
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
        cfg_scale: float = 1.0,
        guidance_fn: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int], Optional[torch.Tensor]]
        ] = None,
        guidance_start: float = 0.8,
        guidance_interval: int = 1,
        guidance_scale: float = 1.0,
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
        if self.cfg.diffusion.cell_init == "cellnet":
            if self.cell_net is None:
                raise ValueError("cellnet requested but not initialized.")
            if cond is None:
                raise ValueError("cellnet requires cond to initialize cell.")
            cell = self.cell_net(cond)
            noise_scale = 0.1 if self.cfg.diffusion.cell_init_noise is None else self.cfg.diffusion.cell_init_noise
            cell = cell + torch.randn_like(cell) * noise_scale
        elif self.cfg.diffusion.cell_rep == "cholesky6" and self.cfg.diffusion.cell_init == "iso":
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
                    guidance_scale=cfg_scale,
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
                    guidance_scale=cfg_scale,
                )
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            if self.cfg.project_each_step:
                frac, cell = self._project_step(frac, cell)
            if self.cfg.project_geometry and uv_angle is not None and z_norm is not None:
                uv_angle, z_norm = self._project_geometry_step(uv_angle, z_norm, atom_mask)
            if self.cfg.z_clamp:
                frac = self._z_clamp_step(frac, cell, atom_mask, slab_t)
            if guidance_fn is not None:
                start_step = int(max(0.0, min(1.0, guidance_start)) * steps)
                if i >= start_step and (i - start_step) % max(guidance_interval, 1) == 0:
                    lattice = self._cell_to_lattice(cell)
                    delta = guidance_fn(frac, lattice, atom_mask, z, i, steps)
                    if delta is not None:
                        frac = frac + float(guidance_scale) * delta
                        frac = frac - torch.floor(frac)
                        frac = frac * atom_mask.unsqueeze(-1)

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

        if self.cfg.project_each_step:
            frac, cell = self._project_step(frac, cell)

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
