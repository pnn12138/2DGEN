"""Placeholder CGCNN implementation suitable for wiring Hydra configs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig


@dataclass
class CGCNNConfig:
    atom_fea_len: int
    h_fea_len: int
    n_conv: int
    cutoff: float
    num_gaussians: int
    pooling: str
    dropout: float
    activation: str
    batch_norm: bool


class CGCNNLightningModule(pl.LightningModule):
    """Thin Lightning wrapper around a CGCNN backbone (to be implemented)."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        hidden = cfg.model.h_fea_len
        self.backbone = nn.Sequential(
            nn.Linear(cfg.model.atom_fea_len, hidden),
            nn.Softplus(),
            nn.Linear(hidden, hidden),
        )
        self.regressor = nn.Linear(hidden, 1)
        self.loss_fn = nn.L1Loss()

    def forward(self, batch):  # pragma: no cover - placeholder
        features = batch["atom_features"]
        hidden = self.backbone(features)
        return self.regressor(hidden).squeeze(-1)

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch)
        loss = self.loss_fn(preds, batch["target"])
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch)
        loss = self.loss_fn(preds, batch["target"])
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.model.optimizer.lr,
            weight_decay=self.hparams.model.optimizer.weight_decay,
        )


def build_cgcnn(cfg: DictConfig) -> CGCNNLightningModule:
    return CGCNNLightningModule(cfg)
