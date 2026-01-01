"""PyTorch Lightning datamodule wiring the Hydra data config to datasets."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from . import download_jdft2d_dataset
from .datasets import Jdft2dDataset, select_ids


class Jdft2dDataModule(pl.LightningDataModule):
    """Creates train/val/test loaders from the cached metadata + split file."""

    def __init__(self, cfg: DictConfig, split_seed: int, feature_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.split_seed = split_seed
        self.feature_dim = feature_dim
        self._metadata = None
        self._splits: Dict[str, List[str]] | None = None

    @property
    def root(self) -> Path:
        return Path(to_absolute_path(str(self.cfg.dataset.root)))

    def prepare_data(self) -> None:
        download_jdft2d_dataset(
            self.root,
            split_seed=self.split_seed,
        )

    def setup(self, stage: str | None = None) -> None:
        metadata_file = (
            self.cfg.dataset.metadata_file
            if "metadata_file" in self.cfg.dataset
            else "jdft2d_exfoliation_metadata.csv"
        )
        metadata_path = self.root / metadata_file
        split_path = self.root / self.cfg.dataset.split_file
        self._metadata = pd.read_csv(metadata_path)
        self._metadata["jid"] = self._metadata["jid"].astype(str)
        self._metadata["atoms"] = self._metadata["atoms"].apply(ast.literal_eval)
        split_data = json.loads(split_path.read_text())
        self._splits = split_data["splits"]

        self.train_dataset = self._build_dataset("train")
        self.val_dataset = self._build_dataset("val")
        self.test_dataset = self._build_dataset("test")

    # pylint: disable=arguments-differ
    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.train_dataset, self.cfg.loader.train)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self.val_dataset, self.cfg.loader.val)

    def test_dataloader(self) -> DataLoader:
        return self._build_loader(self.test_dataset, self.cfg.loader.test)

    def _build_dataset(self, split: str) -> Jdft2dDataset:
        assert self._metadata is not None and self._splits is not None
        ids = select_ids(self._metadata, self._splits[split])
        return Jdft2dDataset(
            self._metadata, ids, self.cfg.dataset.target_key, self.feature_dim
        )

    def _build_loader(self, dataset, loader_cfg) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=loader_cfg.shuffle,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
            drop_last=loader_cfg.drop_last,
        )
