"""Utility to construct a Lightning Trainer from Hydra config."""

from __future__ import annotations

from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig


def build_trainer(cfg: DictConfig) -> pl.Trainer:
    trainer_cfg = cfg.trainer.trainer

    logger = None
    log_cfg = getattr(cfg, "logging", None)
    if log_cfg is not None and getattr(log_cfg.logging, "logger", "") == "tensorboard":
        run_dir = HydraConfig.get().run.dir
        logger = TensorBoardLogger(
            save_dir=str(run_dir),
            name="tb",
            version=None,
            default_hp_metric=False,
        )

    return pl.Trainer(logger=logger, **trainer_cfg)
