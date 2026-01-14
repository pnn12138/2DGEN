from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

from torch.utils.data import Subset


SplitName = Literal["train", "heldout", "all"]


@dataclass(frozen=True)
class C2DBSplit:
    """Train/held-out split indices for a C2DB token cache."""

    schema_version: str
    source_npz: str | None
    seed: int | None
    train_indices: list[int]
    heldout_indices: list[int]


class SplitSubset(Subset):
    """
    torch.utils.data.Subset that forwards attribute access to the underlying dataset.

    Many scripts rely on dataset attributes (e.g. g_scale, geometry fields). A plain
    Subset does not proxy those, so we provide a thin wrapper.
    """

    def __getattr__(self, name: str) -> Any:  # pragma: no cover (simple proxy)
        return getattr(self.dataset, name)


def load_c2db_split(path: Path) -> C2DBSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version") or "c2db_split_v1")
    split = payload.get("split") or {}
    train = split.get("train_indices") or []
    heldout = split.get("heldout_indices") or []
    return C2DBSplit(
        schema_version=schema_version,
        source_npz=payload.get("source_npz"),
        seed=payload.get("seed"),
        train_indices=[int(x) for x in train],
        heldout_indices=[int(x) for x in heldout],
    )


def select_split_indices(split: C2DBSplit, name: SplitName) -> list[int]:
    if name == "train":
        return list(split.train_indices)
    if name == "heldout":
        return list(split.heldout_indices)
    return sorted(set(split.train_indices).union(split.heldout_indices))


def validate_split_indices(indices: Sequence[int], total: int) -> None:
    if any(i < 0 or i >= total for i in indices):
        raise ValueError("Split indices out of range.")
    if len(set(indices)) != len(indices):
        raise ValueError("Split indices contain duplicates.")

