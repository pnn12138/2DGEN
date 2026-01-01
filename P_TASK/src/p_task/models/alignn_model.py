"""Minimal ALIGNN-style regressor for matbench_jdft2d."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from omegaconf import DictConfig


def _aggregate(index: torch.Tensor, src: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class AlignnLayer(nn.Module):
    """One ALIGNN block with line-graph + edge updates."""

    def __init__(self, hidden_dim: int, edge_dim: int, angle_dim: int, dropout: float):
        super().__init__()
        self.line_norm = nn.LayerNorm(edge_dim * 2 + angle_dim)
        self.line_mlp = nn.Sequential(
            nn.Linear(edge_dim * 2 + angle_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_norm = nn.LayerNorm(hidden_dim * 3)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_norm = nn.LayerNorm(hidden_dim * 2)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, node_feats, edge_feats, line_feats, edge_index, line_index):
        l_src, l_dst = line_index
        line_input = torch.cat([edge_feats[l_src], edge_feats[l_dst], line_feats], dim=-1)
        line_messages = self.line_mlp(self.line_norm(line_input))
        line_agg = _aggregate(l_src, line_messages, edge_feats.size(0))

        e_src, e_dst = edge_index
        edge_input = torch.cat([node_feats[e_src], node_feats[e_dst], edge_feats + line_agg], dim=-1)
        edge_messages = self.edge_mlp(self.edge_norm(edge_input))
        edge_feats = edge_feats + edge_messages  # residual

        node_messages = _aggregate(e_dst, edge_feats, node_feats.size(0))
        node_input = torch.cat([node_feats, node_messages], dim=-1)
        node_feats = node_feats + self.node_mlp(self.node_norm(node_input))
        return node_feats, edge_feats


class ALIGNN(nn.Module):
    """Compact ALIGNN backbone."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        max_atomic_num: int,
        dropout: float,
        edge_dim: int,
        angle_dim: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_atomic_num + 1, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.angle_encoder = nn.Sequential(
            nn.Linear(angle_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [AlignnLayer(hidden_dim, hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        node_z = batch["node_feats"].squeeze(-1).clamp(
            min=0, max=self.embedding.num_embeddings - 1
        )
        node_feats = self.embedding(node_z)
        edge_feats = self.edge_encoder(batch["edge_attr"])
        angle_feats = self.angle_encoder(batch["line_attr"])
        for layer in self.layers:
            node_feats, edge_feats = layer(
                node_feats, edge_feats, angle_feats, batch["edge_index"], batch["line_index"]
            )

        # Pool by mean over nodes per graph.
        batch_index = batch["node_batch"]
        pooled = _aggregate(batch_index, node_feats, int(batch_index.max().item()) + 1)
        counts = torch.bincount(batch_index, minlength=pooled.size(0)).float().unsqueeze(-1)
        counts[counts == 0] = 1.0
        pooled = pooled / counts
        return self.readout(pooled).squeeze(-1)


@dataclass
class AlignnConfig:
    hidden_dim: int
    num_layers: int
    dropout: float
    max_atomic_num: int
    edge_dim: int
    angle_dim: int
    target_norm: bool
    target_norm_meta: str
    target_norm: bool
    target_norm_meta: str


class AlignnLightning(pl.LightningModule):
    """Lightning wrapper with MAE loss/metrics."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = ALIGNN(
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            max_atomic_num=cfg.model.max_atomic_num,
            edge_dim=cfg.model.edge_dim,
            angle_dim=cfg.model.angle_dim,
        )
        self.loss_fn = nn.L1Loss()
        self.use_target_norm = bool(cfg.model.get("target_norm", False))
        if self.use_target_norm:
            meta_path = Path(cfg.model.target_norm_meta)
            df = pd.read_csv(meta_path)
            mean = float(df["exfoliation_en"].mean())
            std = float(df["exfoliation_en"].std())
            if math.isclose(std, 0.0):
                std = 1.0
            self.register_buffer("target_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("target_std", torch.tensor(std, dtype=torch.float32))
        else:
            self.register_buffer("target_mean", torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer("target_std", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, batch: dict) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        preds = self.forward(batch)
        target = batch["target"]
        loss = self._compute_loss(preds, target)
        self.log("train/loss", loss, prog_bar=True)
        mae_raw = self._mae_raw(preds, target)
        self.log("train/mae_raw", mae_raw, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.forward(batch)
        target = batch["target"]
        loss = self._compute_loss(preds, target)
        mae_raw = self._mae_raw(preds, target)
        self.log("val/loss", loss, prog_bar=False)
        self.log("val/mae", mae_raw, prog_bar=True)

    def test_step(self, batch, batch_idx):
        preds = self.forward(batch)
        target = batch["target"]
        mae_raw = self._mae_raw(preds, target)
        self.log("test/mae", mae_raw, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.model.optimizer.lr,
            weight_decay=self.hparams.model.optimizer.weight_decay,
        )
        sched_cfg = self.hparams.model.get("scheduler")
        if sched_cfg and sched_cfg.get("name") == "cosine":
            t_max = sched_cfg.get("t_max", None)
            if t_max is None and self.trainer is not None:
                t_max = self.trainer.max_epochs
            t_max = t_max or 100
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=sched_cfg.get("eta_min", 0.0)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        return optimizer

    def _compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_target_norm:
            target = (target - self.target_mean) / self.target_std
            preds = (preds - self.target_mean) / self.target_std
        return self.loss_fn(preds, target)

    def _mae_raw(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_target_norm:
            preds = preds * self.target_std + self.target_mean
        return torch.mean(torch.abs(preds - target))


def build_alignn(cfg: DictConfig) -> AlignnLightning:
    return AlignnLightning(cfg)
