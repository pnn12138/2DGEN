#!/usr/bin/env python
"""Evaluate a trained ALIGNN/CGCNN checkpoint on matbench_jdft2d."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p_task.data.matbench_jdft2d import (  # pylint: disable=wrong-import-position
    MatbenchJdft2dDataModule,
)
from p_task.models import (  # pylint: disable=wrong-import-position
    AlignnLightning,
    CGCNNLightningModule,
    build_alignn,
    build_cgcnn,
)
from p_task.trainer import build_trainer  # pylint: disable=wrong-import-position


CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf"


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="Jdft2d")
def main(cfg: DictConfig) -> None:
    """Run evaluation; set task.checkpoint_path to load a saved model."""
    pl.seed_everything(cfg.task.seed, workers=True)
    model_cfg = cfg.copy()
    if "model" in cfg.model:
        model_cfg.model = cfg.model.model

    datamodule = MatbenchJdft2dDataModule(cfg.data)
    model_name = model_cfg.model.get("name", "")

    checkpoint = getattr(cfg.task, "checkpoint_path", None)
    if checkpoint:
        ckpt_path = Path(checkpoint).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if model_name == "alignn":
            model = AlignnLightning.load_from_checkpoint(str(ckpt_path))
        else:
            model = CGCNNLightningModule.load_from_checkpoint(str(ckpt_path))
    else:
        # Fallback: build fresh model (e.g., to evaluate immediately after training).
        model = build_alignn(model_cfg) if model_name == "alignn" else build_cgcnn(model_cfg)

    trainer = build_trainer(cfg)
    results = trainer.test(model, datamodule=datamodule, verbose=False)
    if results:
        print("Test metrics:", results[0])


if __name__ == "__main__":
    main()
