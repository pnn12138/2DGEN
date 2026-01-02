from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import nn

from twodgen.common.atom_diffusion import AtomDiffusionConfig, AtomVelocityLoss, expand_t, mask_schedule
from twodgen.common.crystal import (
    cholesky6_to_gram6,
    clip_lattice,
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
        nbr_idx: Optional[torch.Tensor] = None,
        nbr_mask: Optional[torch.Tensor] = None,
        dist_nbr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss_fn(self.model, z, frac, atom_mask, gram6, cond, nbr_idx, nbr_mask, dist_nbr)

    def _predict_velocity(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        atom_mask: torch.Tensor,
        gram6: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        cache_every: Optional[int] = None,
        nbr_idx: Optional[torch.Tensor] = None,
        nbr_mask: Optional[torch.Tensor] = None,
        dist_nbr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_v_f, pred_v_g, logits_z = self.model(
            z,
            frac,
            gram6,
            atom_mask,
            t,
            cond,
            step=step,
            cache_every=cache_every,
            nbr_idx=nbr_idx,
            nbr_mask=nbr_mask,
            dist_nbr=dist_nbr,
        )
        return pred_v_f, pred_v_g, logits_z

    @torch.no_grad()
    def _euler_step(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        atom_mask: torch.Tensor,
        gram6: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        cache_every: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_v_f, pred_v_g, logits_z = self._predict_velocity(
            z, frac, atom_mask, gram6, t, cond, step=step, cache_every=cache_every
        )
        delta = expand_t(t_next - t, frac.ndim)
        frac = frac + delta * pred_v_f
        gram6 = gram6 + expand_t(t_next - t, gram6.ndim) * pred_v_g
        return frac, gram6, logits_z

    def _project_step(
        self, frac: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        frac = frac - torch.floor(frac)
        if self.cfg.diffusion.cell_rep == "cholesky6":
            if self.cfg.model.chol_log_min is not None or self.cfg.model.chol_log_max is not None:
                cell = cell.clone()
                cell[:, :3] = torch.clamp(
                    cell[:, :3], min=self.cfg.model.chol_log_min, max=self.cfg.model.chol_log_max
                )
            return frac, cell
        lattice = gram6_to_lattice(cell * self.cfg.model.g_scale)
        cell = lattice_to_gram6(lattice) / self.cfg.model.g_scale
        return frac, cell

    @torch.no_grad()
    def _heun_step(
        self,
        z: torch.Tensor,
        frac: torch.Tensor,
        atom_mask: torch.Tensor,
        gram6: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        cache_every: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_v_f, pred_v_g, logits_z = self._predict_velocity(
            z, frac, atom_mask, gram6, t, cond, step=step, cache_every=cache_every
        )
        delta = expand_t(t_next - t, frac.ndim)
        frac_euler = frac + delta * pred_v_f
        gram_euler = gram6 + expand_t(t_next - t, gram6.ndim) * pred_v_g
        pred_v_f_next, pred_v_g_next, _ = self._predict_velocity(
            z, frac_euler, atom_mask, gram_euler, t_next, cond, step=step, cache_every=cache_every
        )
        pred_v_f = 0.5 * (pred_v_f + pred_v_f_next)
        pred_v_g = 0.5 * (pred_v_g + pred_v_g_next)
        frac = frac + delta * pred_v_f
        gram6 = gram6 + expand_t(t_next - t, gram6.ndim) * pred_v_g
        return frac, gram6, logits_z

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        method = method or self.cfg.sampling_method
        steps = steps if steps is not None else self.cfg.num_sampling_steps
        device = self.model.z_embed.weight.device

        atom_mask = torch.zeros(batch_size, max_atoms, device=device)
        atom_mask[:, :num_atoms] = 1.0

        z = torch.zeros(batch_size, max_atoms, dtype=torch.long, device=device)
        z[:, :num_atoms] = self.model.mask_id

        frac = torch.randn(batch_size, max_atoms, 3, device=device) * self.cfg.diffusion.noise_scale_f
        frac = frac * atom_mask.unsqueeze(-1)
        if self.cfg.diffusion.cell_rep == "cholesky6" and self.cfg.diffusion.cell_init == "iso":
            scale = 1.0 if self.cfg.diffusion.cell_init_scale is None else self.cfg.diffusion.cell_init_scale
            log_s = torch.log(torch.tensor(scale, device=device))
            if self.cfg.model.chol_log_min is not None or self.cfg.model.chol_log_max is not None:
                log_s = torch.clamp(log_s, min=self.cfg.model.chol_log_min, max=self.cfg.model.chol_log_max)
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
        for i in range(steps):
            t = t_schedule[i].expand(batch_size)
            t_next = t_schedule[i + 1].expand(batch_size)
            if method == "euler":
                frac, cell, logits_z = self._euler_step(
                    z,
                    frac,
                    atom_mask,
                    cell,
                    t,
                    t_next,
                    cond,
                    step=i,
                    cache_every=self.cfg.neighbor_update_steps,
                )
            elif method == "heun":
                frac, cell, logits_z = self._heun_step(
                    z,
                    frac,
                    atom_mask,
                    cell,
                    t,
                    t_next,
                    cond,
                    step=i,
                    cache_every=self.cfg.neighbor_update_steps,
                )
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            if self.cfg.project_each_step:
                frac, cell = self._project_step(frac, cell)

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
                z_pred = sample_z(logits[reveal_idx])
                z[b, reveal_idx] = z_pred

        frac = frac - torch.floor(frac)
        if self.cfg.diffusion.cell_rep == "cholesky6":
            gram6 = cholesky6_to_gram6(
                cell, log_min=self.cfg.model.chol_log_min, log_max=self.cfg.model.chol_log_max
            )
        else:
            gram6 = cell
        return z, frac, gram6, atom_mask

    @torch.no_grad()
    def gram6_to_lattice(self, gram6: torch.Tensor) -> torch.Tensor:
        lattice = gram6_to_lattice(gram6 * self.cfg.model.g_scale)
        if self.cfg.reduce_lattice:
            lattice = reduce_lattice_simple(lattice)
        if self.cfg.niggli_reduce:
            lattice = niggli_reduce_lattice(lattice)
        return lattice


__all__ = ["AtomDenoiserConfig", "AtomDenoiser"]
