"""Dataset objects wrapping the downloaded metadata."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from torch.utils.data import Dataset


class Jdft2dDataset(Dataset):
    """Dataset that emits lightweight CGCNN-style feature tensors."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        ids: Sequence[str],
        target_key: str,
        feature_dim: int,
    ) -> None:
        self._metadata = metadata.set_index("jid")
        self._ids = list(ids)
        self._target_key = target_key
        self._feature_dim = feature_dim

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._ids)

    def __getitem__(self, idx: int):
        jid = self._ids[idx]
        row = self._metadata.loc[jid]
        target = float(row[self._target_key])
        atoms_dict = row["atoms"]
        features = self._featurize(atoms_dict)
        return {
            "jid": jid,
            "atom_features": features,
            "target": torch.tensor(target, dtype=torch.float32),
        }

    def _featurize(self, atoms_dict) -> torch.Tensor:
        atoms = Atoms.from_dict(atoms_dict)
        feature_vec = torch.zeros(self._feature_dim, dtype=torch.float32)
        components = self._scalar_features(atoms)
        feature_vec[: len(components)] = torch.as_tensor(components[: self._feature_dim])
        return feature_vec

    @staticmethod
    def _scalar_features(atoms: Atoms) -> Sequence[float]:
        atomic_numbers = np.array(atoms.atomic_numbers, dtype=np.float32)
        stats = [
            float(atoms.num_atoms),
            float(atoms.volume),
            float(atoms.density),
        ]
        if len(atomic_numbers):
            stats.extend(
                [
                    float(np.mean(atomic_numbers)),
                    float(np.std(atomic_numbers)),
                    float(np.min(atomic_numbers)),
                    float(np.max(atomic_numbers)),
                ]
            )
        else:
            stats.extend([0.0, 0.0, 0.0, 0.0])
        lengths = atoms.lattice.abc
        angles = atoms.lattice.angles
        stats.extend([float(lengths[0]), float(lengths[1]), float(lengths[2])])
        stats.extend([float(angles[0]), float(angles[1]), float(angles[2])])
        return stats


def select_ids(metadata: pd.DataFrame, ids: Iterable[str]):
    available = set(metadata["jid"].astype(str).tolist())
    ordered = [jid for jid in ids if str(jid) in available]
    return ordered
