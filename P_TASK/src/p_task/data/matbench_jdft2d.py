"""Dataset + datamodule for matbench_jdft2d ALIGNN baseline."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .graphs_alignn import GraphData, build_alignn_graph, collate_graphs
from .prepare_jdft2d_cache import DEFAULT_CACHE_DIR, build_cache


@dataclass
class GraphConfig:
    cutoff: float
    max_neighbors: int
    num_rbf: int
    num_abf: int


class MatbenchJdft2dDataset(Dataset):
    """Wrap cached structures + metadata and emit ALIGNN graph dicts."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        structures: Sequence,
        sample_ids: Sequence[str],
        graph_cfg: GraphConfig,
    ) -> None:
        self.metadata = metadata.set_index("sample_id")
        self.structures = structures
        self.sample_ids = list(sample_ids)
        self.graph_cfg = graph_cfg

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sample_id = self.sample_ids[idx]
        row = self.metadata.loc[sample_id]
        struct_idx = int(sample_id.split("-")[-1]) - 1
        structure = self.structures[struct_idx]
        graph: GraphData = build_alignn_graph(
            structure,
            cutoff=self.graph_cfg.cutoff,
            max_neighbors=self.graph_cfg.max_neighbors,
            num_rbf=self.graph_cfg.num_rbf,
            num_abf=self.graph_cfg.num_abf,
        )
        return {
            "sample_id": sample_id,
            "graph": graph,
            "target": torch.tensor(float(row["exfoliation_en"]), dtype=torch.float32),
        }


class MatbenchJdft2dDataModule(pl.LightningDataModule):
    """Lightning data module providing train/val/test loaders."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cache_dir = Path(cfg.dataset.root)
        self.fold = cfg.dataset.fold
        self.val_fraction = cfg.dataset.val_fraction
        self._metadata = None
        self._structures = None
        self._splits = None

    def prepare_data(self) -> None:
        build_cache(self.cache_dir, force=self.cfg.dataset.get("force_download", False))

    def setup(self, stage: str | None = None) -> None:
        meta_path = self.cache_dir / "jdft2d_meta.csv"
        struct_path = self.cache_dir / "structures.pkl"
        splits_path = Path(self.cfg.dataset.splits_file)
        self._metadata = pd.read_csv(meta_path)
        with open(struct_path, "rb") as f:
            self._structures = pickle.load(f)
        splits_raw = json.loads(Path(splits_path).read_text())
        self._splits = splits_raw["folds"][f"fold_{self.fold}"]

        graph_cfg = GraphConfig(
            cutoff=self.cfg.graph.cutoff,
            max_neighbors=self.cfg.graph.max_neighbors,
            num_rbf=self.cfg.graph.num_rbf,
            num_abf=self.cfg.graph.num_abf,
        )
        train_ids = self._splits["train"]
        test_ids = self._splits["test"]
        train_ids, val_ids = _train_val_split(
            train_ids, val_fraction=self.val_fraction, seed=self.cfg.dataset.seed
        )
        self.train_dataset = MatbenchJdft2dDataset(
            self._metadata, self._structures, train_ids, graph_cfg
        )
        self.val_dataset = MatbenchJdft2dDataset(
            self._metadata, self._structures, val_ids, graph_cfg
        )
        self.test_dataset = MatbenchJdft2dDataset(
            self._metadata, self._structures, test_ids, graph_cfg
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.loader.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.loader.train.num_workers,
            collate_fn=_collate,
        )

    def val_dataloader(self) -> DataLoader:
        if len(self.val_dataset) == 0:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.loader.val.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.val.num_workers,
            collate_fn=_collate,
        )

    def test_dataloader(self) -> DataLoader:
        if len(self.test_dataset) == 0:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.loader.test.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.test.num_workers,
            collate_fn=_collate,
        )


def _train_val_split(ids: Sequence[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = torch.Generator().manual_seed(seed)
    ids_tensor = torch.randperm(len(ids), generator=rng)
    if val_fraction <= 0:
        return list(ids), []
    n_val = max(1, int(len(ids) * val_fraction))
    val_idx = ids_tensor[:n_val].tolist()
    train_idx = ids_tensor[n_val:].tolist()
    ids_list = list(ids)
    train_ids = [ids_list[i] for i in train_idx]
    val_ids = [ids_list[i] for i in val_idx]
    return train_ids, val_ids


def _collate(batch: List[dict]) -> dict:
    graphs = [item["graph"] for item in batch]
    merged = collate_graphs(graphs)
    targets = torch.stack([item["target"] for item in batch], dim=0)
    merged["target"] = targets
    merged["sample_id"] = [item["sample_id"] for item in batch]
    return merged
