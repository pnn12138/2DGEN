from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from torch import nn

from model.model import C2DBJiT, JiTC2DBConfig
from common.diffusion import CleanPredictionConfig, VelocityPredictionLoss, drop_labels, expand_t, logit_normal_sample


@dataclass
class DenoiserConfig:
    """
    Wrapper config to bundle model + diffusion hyperparameters.
    """

    model: JiTC2DBConfig = field(default_factory=JiTC2DBConfig)
    diffusion: CleanPredictionConfig = field(default_factory=CleanPredictionConfig)

    # generation / sampling
    sampling_method: str = "euler"
    num_sampling_steps: int = 20
    cfg_scale: float = 1.0
    cfg_interval: Tuple[float, float] = (0.0, 1.0)


class C2DBDenoiser(nn.Module):
    """
    Thin wrapper separating model and loss, mirroring JiT's structure but sized for 3x24x3.

    - model: predicts clean x0 given noisy z and timestep t.
    - loss: velocity-prediction loss (v-space) computed via VelocityPredictionLoss.
    """

    def __init__(self, cfg: DenoiserConfig = DenoiserConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = C2DBJiT(cfg.model)
        self.loss_fn = VelocityPredictionLoss(cfg.diffusion)
        self.in_chans = cfg.model.in_chans
        self.img_size = cfg.model.img_size

    def forward(self, x0: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Train step: returns (loss, x_pred, t).
        """
        return self.loss_fn(self.model, x0, labels)

    def predict_clean(self, z: torch.Tensor, t: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Expose raw model forward for inference/sampling pipelines."""
        return self.model(z, t, labels)

    # ---------- Sampling helpers ----------
    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Logit-normal timestep sampling aligned with training distribution."""
        diff_cfg = self.cfg.diffusion
        return logit_normal_sample(batch_size, device, diff_cfg.P_mean, diff_cfg.P_std)

    def _predict_velocity(self, z: torch.Tensor, t: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute v_pred = (x_pred - z) / (1 - t) with stabilization.
        """
        diff_cfg = self.cfg.diffusion
        t_expand = expand_t(t, z.ndim)
        x_pred = self.model(z, t, labels)
        denom = (1.0 - t_expand).clamp_min(diff_cfg.t_eps)
        return (x_pred - z) / denom

    def _cfg_velocity(
        self, z: torch.Tensor, t: torch.Tensor, labels: Optional[torch.Tensor], cfg_scale: float, cfg_interval: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Classifier-free guidance in velocity space. Falls back to unconditional if labels are absent.
        """
        num_classes = self.cfg.diffusion.num_classes
        if labels is None or num_classes is None:
            return self._predict_velocity(z, t, labels)

        v_cond = self._predict_velocity(z, t, labels)
        v_uncond = self._predict_velocity(z, t, torch.full_like(labels, num_classes))

        low, high = cfg_interval
        interval_mask = (t < high) & ((low == 0.0) | (t > low))
        scale = torch.where(interval_mask, torch.full_like(t, cfg_scale), torch.ones_like(t))
        scale_expand = expand_t(scale, z.ndim)
        return v_uncond + scale_expand * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: Optional[torch.Tensor],
        cfg_scale: float,
        cfg_interval: Tuple[float, float],
    ) -> torch.Tensor:
        v_pred = self._cfg_velocity(z, t, labels, cfg_scale, cfg_interval)
        return z + (expand_t(t_next, z.ndim) - expand_t(t, z.ndim)) * v_pred

    @torch.no_grad()
    def _heun_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: Optional[torch.Tensor],
        cfg_scale: float,
        cfg_interval: Tuple[float, float],
    ) -> torch.Tensor:
        v_pred_t = self._cfg_velocity(z, t, labels, cfg_scale, cfg_interval)
        z_euler = z + (expand_t(t_next, z.ndim) - expand_t(t, z.ndim)) * v_pred_t
        v_pred_t_next = self._cfg_velocity(z_euler, t_next, labels, cfg_scale, cfg_interval)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (expand_t(t_next, z.ndim) - expand_t(t, z.ndim)) * v_pred

    @torch.no_grad()
    def generate(
        self,
        labels: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        method: Optional[str] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        cfg_interval: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """
        ODE-style sampler supporting Euler/Heun, adapted to 3x24x3 grids.

        Args:
            labels: optional (B,) tensor. If None, falls back to zeros when num_classes is not set.
            batch_size: required when labels are None.
        """
        method = method or self.cfg.sampling_method
        steps = steps if steps is not None else self.cfg.num_sampling_steps
        cfg_scale = cfg_scale if cfg_scale is not None else self.cfg.cfg_scale
        cfg_interval = cfg_interval if cfg_interval is not None else self.cfg.cfg_interval

        device = labels.device if isinstance(labels, torch.Tensor) else self.model.pos_embed.device
        if labels is None:
            if batch_size is None:
                raise ValueError("batch_size is required when labels are None.")
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        bsz = labels.size(0)

        # optional label dropout for conditional runs (classifier-free guidance)
        if self.cfg.diffusion.num_classes is not None and self.training:
            labels = drop_labels(labels, self.cfg.diffusion.label_drop_prob, self.cfg.diffusion.num_classes)

        z = torch.randn(bsz, self.in_chans, *self.img_size, device=device) * self.cfg.diffusion.noise_scale
        t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)

        if method == "euler":
            step_fn = self._euler_step
        elif method == "heun":
            step_fn = self._heun_step
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        for i in range(steps):
            t = t_schedule[i].expand(bsz)
            t_next = t_schedule[i + 1].expand(bsz)
            z = step_fn(z, t, t_next, labels, cfg_scale, cfg_interval)

        return z


__all__ = ["DenoiserConfig", "C2DBDenoiser"]
