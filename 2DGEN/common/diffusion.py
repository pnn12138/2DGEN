from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class CleanPredictionConfig:
    """
    Hyper-parameters for direct-clean (x_0) prediction, matching JiT's noise setup.

    P_mean/P_std: control logit-normal sampling of timestep t in (0,1).
    t_eps: clamp to avoid division by zero.
    noise_scale: std of Gaussian noise added before predicting x_0.
    label_drop_prob: classifier-free guidance dropout on labels (if used).
    """

    P_mean: float = -1.2
    P_std: float = 1.2
    t_eps: float = 1e-5
    noise_scale: float = 1.0
    label_drop_prob: float = 0.1
    num_classes: Optional[int] = None  # set to enable label dropping


def logit_normal_sample(batch_size: int, device: torch.device, P_mean: float, P_std: float) -> torch.Tensor:
    """Sample t ~ sigmoid(N(P_mean, P_std^2))."""
    z = torch.randn(batch_size, device=device) * P_std + P_mean
    return torch.sigmoid(z)


def expand_t(t: torch.Tensor, ndims: int) -> torch.Tensor:
    """Expand timestep tensor to match input dimensionality."""
    return t.view(-1, *([1] * (ndims - 1)))


def drop_labels(labels: torch.Tensor, drop_prob: float, num_classes: int) -> torch.Tensor:
    """Classifier-free guidance style label dropout."""
    drop = torch.rand(labels.shape[0], device=labels.device) < drop_prob
    return torch.where(drop, torch.full_like(labels, num_classes), labels)


class CleanPredictionLoss(nn.Module):
    """
    Computes L2 loss between predicted x_0 and clean target, using JiT-style noisy inputs.

    Expected model signature: model(z, t_flat, labels=None) -> x_pred with same shape as z.
    """

    def __init__(self, cfg: CleanPredictionConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            model: network predicting clean sample x_0 from noisy input.
            x0: clean target, shape (B, C, H, W) e.g., (B, 3, 24, 3).
            labels: optional class labels for conditional training.
        Returns:
            loss: scalar tensor.
            x_pred: predicted clean sample.
            t: sampled timesteps (B,).
        """
        device = x0.device
        bsz = x0.size(0)

        t = logit_normal_sample(bsz, device, self.cfg.P_mean, self.cfg.P_std)
        t_expand = expand_t(t, x0.ndim)

        noise = torch.randn_like(x0) * self.cfg.noise_scale
        z = t_expand * x0 + (1.0 - t_expand) * noise

        labels_in = labels
        if labels is not None and self.cfg.num_classes is not None:
            labels_in = drop_labels(labels, self.cfg.label_drop_prob, self.cfg.num_classes)

        x_pred = model(z, t, labels_in)
        loss = (x_pred - x0) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss, x_pred, t


class VelocityPredictionLoss(nn.Module):
    """
    Velocity (v-pred) loss mirroring JiT: model输出干净样本 x_pred，损失在 v 空间计算。
    v = (x0 - z) / (1 - t)，v_pred = (x_pred - z) / (1 - t)
    """

    def __init__(self, cfg: CleanPredictionConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x0.device
        bsz = x0.size(0)

        t = logit_normal_sample(bsz, device, self.cfg.P_mean, self.cfg.P_std)
        t_expand = expand_t(t, x0.ndim)

        noise = torch.randn_like(x0) * self.cfg.noise_scale
        z = t_expand * x0 + (1.0 - t_expand) * noise

        labels_in = labels
        if labels is not None and self.cfg.num_classes is not None:
            labels_in = drop_labels(labels, self.cfg.label_drop_prob, self.cfg.num_classes)

        x_pred = model(z, t, labels_in)

        denom = (1.0 - t_expand).clamp_min(self.cfg.t_eps)
        v = (x0 - z) / denom
        v_pred = (x_pred - z) / denom

        loss = (v - v_pred) ** 2
        loss = loss.mean(dim=(1, 2, 3)).mean()

        return loss, x_pred, t


__all__ = ["CleanPredictionConfig", "CleanPredictionLoss", "VelocityPredictionLoss"]
