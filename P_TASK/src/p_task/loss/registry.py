"""Simple loss/metric factory used by the training loop skeleton."""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig


def build_loss(cfg: DictConfig) -> nn.Module:
    name = cfg.loss.name.lower()
    if name == "l1":
        return nn.L1Loss(reduction=cfg.loss.reduction)
    if name == "l2":
        return nn.MSELoss(reduction=cfg.loss.reduction)
    raise ValueError(f"Unsupported loss type: {cfg.loss.name}")
