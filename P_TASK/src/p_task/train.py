"""Hydra entry for ALIGNN/CGCNN experiments on matbench_jdft2d."""

from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from p_task.data.matbench_jdft2d import MatbenchJdft2dDataModule  # pylint: disable=wrong-import-position
from p_task.models import build_alignn, build_cgcnn  # pylint: disable=wrong-import-position
from p_task.trainer import build_trainer  # pylint: disable=wrong-import-position

CONFIG_DIR = Path(__file__).resolve().parents[2] / "conf"

@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="Jdft2d")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.task.seed, workers=True)
    # Unwrap model config (alignn config nests under model.model)
    model_cfg = cfg.copy()
    if "model" in cfg.model:
        model_cfg.model = cfg.model.model

    if cfg.data.dataset.name == "JARVIS_Jdft2d":
        from p_task.data.datamodule import Jdft2dDataModule  # lazy import to avoid jarvis deps

        datamodule = Jdft2dDataModule(
            cfg.data, split_seed=cfg.task.seed, feature_dim=model_cfg.model.atom_fea_len
        )
    else:
        datamodule = MatbenchJdft2dDataModule(cfg.data)

    model_name = model_cfg.model.get("name", "")
    model = build_alignn(model_cfg) if model_name == "alignn" else build_cgcnn(model_cfg)
    trainer = build_trainer(cfg)
    trainer.fit(model, datamodule=datamodule)
    test_results = trainer.test(model, datamodule=datamodule, verbose=False)
    if test_results:
        print("Test metrics:", test_results[0])


if __name__ == "__main__":
    main()
