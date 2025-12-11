"""Hydra entry script for Jdft2d + CGCNN training."""

from __future__ import annotations

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from .data.datamodule import Jdft2dDataModule
from .models import build_cgcnn
from .trainer import build_trainer


@hydra.main(version_base=None, config_path="../../conf", config_name="Jdft2d")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.task.seed, workers=True)
    datamodule = Jdft2dDataModule(
        cfg.data,
        split_seed=cfg.task.seed,
        feature_dim=cfg.model.atom_fea_len,
    )
    model = build_cgcnn(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
