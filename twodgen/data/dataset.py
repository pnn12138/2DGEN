from __future__ import annotations

import warnings
from pathlib import Path

from .c2db_dataset import C2DBTokenNPZDataset


class CrystDataset(C2DBTokenNPZDataset):
    """
    Backwards-compatible alias for token-cache datasets.

    Historically this module contained an unfinished stub and a hard dependency on
    `torch_geometric`. The twodgen token pipeline uses `C2DBTokenNPZDataset` instead.
    """

    def __init__(self, path: str | Path) -> None:
        warnings.warn(
            "`data.dataset.CrystDataset` is deprecated; use `data.c2db_dataset.C2DBTokenNPZDataset`.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(path)


__all__ = ["CrystDataset"]
