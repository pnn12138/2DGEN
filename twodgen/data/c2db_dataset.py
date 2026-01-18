from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure
from torch.utils.data import Dataset


def _coerce_optional_float(value: object) -> Optional[float]:
    """Convert NaN to None and keep real numbers."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        # Non-numeric values should just be returned verbatim
        return None
    return float(value)


def _pad_1d(
    values: np.ndarray, max_len: int, pad_value: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad or truncate a 1D array and build a mask of valid entries."""
    trimmed = values[:max_len]
    padded = np.full((max_len,), pad_value, dtype=trimmed.dtype)
    padded[: len(trimmed)] = trimmed

    mask = np.zeros((max_len,), dtype=np.float32)
    mask[: len(trimmed)] = 1.0
    return padded, mask


def _pad_2d(
    values: np.ndarray, max_len: int, pad_value: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad or truncate a (N, 3) array and build a mask of valid rows."""
    trimmed = values[:max_len]
    padded = np.full((max_len, 3), pad_value, dtype=np.float32)
    padded[: len(trimmed)] = trimmed.astype(np.float32)

    mask = np.zeros((max_len,), dtype=np.float32)
    mask[: len(trimmed)] = 1.0
    return padded, mask


@dataclass
class C2DBMetadata:
    """Lightweight container for per-entry metadata."""

    material_id: str
    chemical_formula: Optional[str]
    space_group_number: Optional[float]
    space_group_symbol: Optional[str]
    total_energy: Optional[float]
    formation_energy: Optional[float]
    energy_above_hull: Optional[float]
    exfoliation_energy: Optional[float]


class C2DBDataset(Dataset):
    """
    Dataset that reads entries from the C2DB summary CSV and returns padded tensors.

    Each item keeps atom types and fractional coordinates aligned by index, with an
    `atom_mask` indicating which positions are real atoms (1) versus padding (0).
    """

    def __init__(
        self,
        csv_path: Path | str,
        max_atoms: int = 24,
        pad_value: float = 0.0,
        drop_if_too_many_atoms: bool = True,
        limit: Optional[int] = None,
    ) -> None:
        """
        Args:
            csv_path: Path to `c2db_summary.csv`.
            max_atoms: Maximum atoms kept per structure; samples with more atoms
                are dropped when `drop_if_too_many_atoms` is True.
            pad_value: Value used for padding atom numbers and fractional coords.
            drop_if_too_many_atoms: Whether to skip rows exceeding `max_atoms`.
            limit: Optional cap on number of rows to load (useful for quick tests).
        """
        self.csv_path = Path(csv_path)
        self.max_atoms = max_atoms
        self.pad_value = pad_value
        self.drop_if_too_many_atoms = drop_if_too_many_atoms
        self.limit = limit

        self.samples: List[Dict[str, torch.Tensor]] = []
        self.metadata: List[C2DBMetadata] = []

        self._load()

    def _load(self) -> None:
        df = pd.read_csv(self.csv_path)
        if self.limit is not None:
            df = df.head(self.limit)

        for row in df.itertuples(index=False):
            cif: Optional[str] = getattr(row, "cif", None)
            if not isinstance(cif, str) or not cif.strip():
                continue

            try:
                structure = Structure.from_str(cif, fmt="cif")
            except Exception:
                # Skip malformed entries to keep downstream code simple
                continue

            num_atoms = len(structure)
            if num_atoms > self.max_atoms and self.drop_if_too_many_atoms:
                continue

            atomic_numbers = np.array(
                [site.specie.number for site in structure.sites], dtype=np.int64
            )
            frac_coords = np.asarray(structure.frac_coords, dtype=np.float32)
            lattice_matrix = np.asarray(structure.lattice.matrix, dtype=np.float32)

            padded_numbers, mask_numbers = _pad_1d(
                atomic_numbers, self.max_atoms, self.pad_value
            )
            padded_coords, mask_coords = _pad_2d(
                frac_coords, self.max_atoms, self.pad_value
            )

            # Consistency check: both masks should agree
            atom_mask = np.minimum(mask_numbers, mask_coords)

            sample = {
                "atomic_numbers": torch.from_numpy(padded_numbers),
                "frac_coords": torch.from_numpy(padded_coords),
                "atom_mask": torch.from_numpy(atom_mask),
                "lattice_matrix": torch.from_numpy(lattice_matrix),
            }
            self.samples.append(sample)

            meta = C2DBMetadata(
                material_id=getattr(row, "material_id"),
                chemical_formula=getattr(row, "chemical_formula", None),
                space_group_number=_coerce_optional_float(
                    getattr(row, "space_group_number", None)
                ),
                space_group_symbol=getattr(row, "space_group_symbol", None),
                total_energy=_coerce_optional_float(getattr(row, "total_energy", None)),
                formation_energy=_coerce_optional_float(
                    getattr(row, "formation_energy", None)
                ),
                energy_above_hull=_coerce_optional_float(
                    getattr(row, "energy_above_hull", None)
                ),
                exfoliation_energy=_coerce_optional_float(
                    getattr(row, "exfoliation_energy", None)
                ),
            )
            self.metadata.append(meta)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]

    def get_metadata(self, idx: int) -> C2DBMetadata:
        return self.metadata[idx]

    @staticmethod
    def collate_fn(
        batch: Sequence[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Collate samples into a batch of tensors."""
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


class C2DBAtomDataset(Dataset):
    """
    Dataset for token-based diffusion: returns padded Z/F/mask plus scaled Gram-6 g.
    """

    def __init__(
        self,
        csv_path: Path | str,
        max_atoms: int = 24,
        pad_value: float = 0.0,
        drop_if_too_many_atoms: bool = True,
        limit: Optional[int] = None,
        g_scale: float = 100.0,
        niggli_reduce: bool = False,
    ) -> None:
        self.base = C2DBDataset(
            csv_path=csv_path,
            max_atoms=max_atoms,
            pad_value=pad_value,
            drop_if_too_many_atoms=drop_if_too_many_atoms,
            limit=limit,
        )
        self.g_scale = g_scale
        self.niggli_reduce = niggli_reduce

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base[idx]
        if self.niggli_reduce:
            try:
                atom_mask = sample["atom_mask"] > 0.5
                atomic_numbers = sample["atomic_numbers"][atom_mask].cpu().numpy().astype(np.int64)
                frac_coords = sample["frac_coords"][atom_mask].cpu().numpy().astype(np.float32)
                lattice = sample["lattice_matrix"].cpu().numpy().astype(np.float32)
                if atomic_numbers.size > 0:
                    structure = Structure(
                        lattice=lattice,
                        species=atomic_numbers.tolist(),
                        coords=frac_coords,
                        coords_are_cartesian=False,
                    )
                    reduced = structure.get_reduced_structure("niggli")
                    atomic_numbers = np.array(
                        [site.specie.number for site in reduced.sites],
                        dtype=np.int64,
                    )
                    frac_coords = np.asarray(reduced.frac_coords, dtype=np.float32)
                    lattice = np.asarray(reduced.lattice.matrix, dtype=np.float32)
                    padded_numbers, mask_numbers = _pad_1d(
                        atomic_numbers, self.base.max_atoms, self.base.pad_value
                    )
                    padded_coords, mask_coords = _pad_2d(
                        frac_coords, self.base.max_atoms, self.base.pad_value
                    )
                    atom_mask = np.minimum(mask_numbers, mask_coords)
                    sample = {
                        "atomic_numbers": torch.from_numpy(padded_numbers),
                        "frac_coords": torch.from_numpy(padded_coords),
                        "atom_mask": torch.from_numpy(atom_mask),
                        "lattice_matrix": torch.from_numpy(lattice),
                    }
            except Exception:
                # Fall back to original sample if Niggli reduction fails.
                pass
        lattice = sample["lattice_matrix"].float()
        g = _lattice_to_gram6(lattice) / self.g_scale
        counts_vector = _counts_vector(sample["atomic_numbers"], sample["atom_mask"])
        return {
            "atomic_numbers": sample["atomic_numbers"].long(),
            "frac_coords": sample["frac_coords"].float(),
            "atom_mask": sample["atom_mask"].float(),
            "gram6": g.float(),
            "counts_vector": counts_vector,
        }

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


class C2DBTokenNPZDataset(Dataset):
    """
    Dataset that reads token caches saved by prepare_c2db_tokens.py.
    """

    def __init__(self, npz_path: Path | str, align_atoms: bool = True, coord_frame: str = "raw") -> None:
        super().__init__()
        data = np.load(npz_path)
        gram6_convention = data["gram6_convention"].item() if "gram6_convention" in data else None
        if gram6_convention is None:
            raise ValueError(
                "Legacy token cache detected (missing `gram6_convention`). "
                "This repo now assumes row-vector lattices (cart = frac @ lattice) and "
                "Gram6 computed from `G = lattice @ lattice^T`. "
                "Please re-run preprocessing or migrate the cache via "
                "`uv run python -m twodgen.data.migrate_gram6_convention --in <old.npz> --out <new.npz>`."
            )
        if str(gram6_convention) != "row_lattice":
            raise ValueError(f"Unsupported gram6_convention={gram6_convention!r} (expected 'row_lattice').")
        self.z = torch.from_numpy(data["z"]).long()
        self.f = torch.from_numpy(data["f"]).float()
        self.atom_mask = torch.from_numpy(data["atom_mask"]).float()
        self.f_canon = torch.from_numpy(data["f_canon"]).float() if "f_canon" in data else None
        self.atom_mask_canon = (
            torch.from_numpy(data["atom_mask_canon"]).float() if "atom_mask_canon" in data else None
        )
        self.gram6 = torch.from_numpy(data["gram6"]).float()
        self.lattice = torch.from_numpy(data["lattice"]).float() if "lattice" in data else None
        self.gram6_canon = torch.from_numpy(data["gram6_canon"]).float() if "gram6_canon" in data else None
        self.lattice_canon = torch.from_numpy(data["lattice_canon"]).float() if "lattice_canon" in data else None
        self.material_ids = data["material_id"].tolist() if "material_id" in data else None
        self.max_atoms = int(data["max_atoms"]) if "max_atoms" in data else self.z.shape[1]
        self.g_scale = float(data["g_scale"]) if "g_scale" in data else 1.0
        self.z_canon = torch.from_numpy(data["z_canon"]).long() if "z_canon" in data else None
        self.uvz = torch.from_numpy(data["uvz"]).float() if "uvz" in data else None
        self.uv_angle = torch.from_numpy(data["uv_angle"]).float() if "uv_angle" in data else None
        self.u = torch.from_numpy(data["u"]).float() if "u" in data else None
        self.v = torch.from_numpy(data["v"]).float() if "v" in data else None
        self.z_norm = torch.from_numpy(data["z_norm"]).float() if "z_norm" in data else None
        self.t = torch.from_numpy(data["t"]).float() if "t" in data else None
        self.a_hat = torch.from_numpy(data["a_hat"]).float() if "a_hat" in data else None
        self.b_hat = torch.from_numpy(data["b_hat"]).float() if "b_hat" in data else None
        self.n = torch.from_numpy(data["n"]).float() if "n" in data else None
        self.lattice_param = torch.from_numpy(data["lattice_param"]).float() if "lattice_param" in data else None
        self.counts_vector = (
            torch.from_numpy(data["counts_vector"]).long() if "counts_vector" in data else None
        )
        self.order_idx = torch.from_numpy(data["order_idx"]).long() if "order_idx" in data else None
        self.order_inv = torch.from_numpy(data["order_inv"]).long() if "order_inv" in data else None
        self.cond_t_mean = torch.from_numpy(data["cond_t_mean"]).float() if "cond_t_mean" in data else None
        self.cond_t_std = torch.from_numpy(data["cond_t_std"]).float() if "cond_t_std" in data else None
        self.schema_version = data["schema_version"].item() if "schema_version" in data else None
        self.coord_frame = data["coord_frame"].item() if "coord_frame" in data else None
        self.align_atoms = align_atoms
        self.coord_frame_requested = coord_frame
        self.coord_frame_actual = coord_frame
        if coord_frame == "canon" and (self.f_canon is None or self.gram6_canon is None):
            self.coord_frame_actual = "raw"
        known = {
            "z",
            "f",
            "atom_mask",
            "f_canon",
            "atom_mask_canon",
            "lattice",
            "gram6",
            "gram6_canon",
            "gram6_convention",
            "gram6_version",
            "material_id",
            "max_atoms",
            "g_scale",
            "schema_version",
            "coord_frame",
            "z_canon",
            "lattice_canon",
            "uvz",
            "uv_angle",
            "u",
            "v",
            "z_norm",
            "t",
            "a_hat",
            "b_hat",
            "n",
            "lattice_param",
            "counts_vector",
            "order_idx",
            "order_inv",
            # Deprecated neighbor caches (kept in `known` so legacy npz loads cleanly).
            "nbr_idx",
            "nbr_dist",
            "nbr_mask",
            "preprocess_v3",
            "preprocess_version",
            "eps_area",
            "eps_inv",
            "round_prec",
            "z_norm_clip",
            "neighbor_k",
            "cond_lattice_mean",
            "cond_lattice_std",
            "cond_t_mean",
            "cond_t_std",
        }
        self.extra: Dict[str, torch.Tensor] = {}
        for key in data.files:
            if key in known:
                continue
            value = np.asarray(data[key])
            if value.ndim == 0 or value.shape[0] != self.z.shape[0]:
                continue
            self.extra[key] = torch.from_numpy(value)

    def __len__(self) -> int:
        return self.z.shape[0]

    @staticmethod
    def _invert_order_idx(order_idx: torch.Tensor) -> torch.Tensor:
        order_inv = torch.full_like(order_idx, -1)
        valid = order_idx >= 0
        if not torch.any(valid):
            return order_inv
        order_inv[order_idx[valid]] = torch.arange(valid.sum(), device=order_idx.device, dtype=order_idx.dtype)
        return order_inv

    @staticmethod
    def _reorder_by_index(values: torch.Tensor, index: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        out = torch.zeros_like(values)
        if pad_value != 0.0:
            out.fill_(pad_value)
        valid = index >= 0
        if torch.any(valid):
            out[valid] = values[index[valid]]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        order_idx = self.order_idx[idx] if self.order_idx is not None else None
        order_inv = self.order_inv[idx] if self.order_inv is not None else None
        if order_inv is None and order_idx is not None:
            order_inv = self._invert_order_idx(order_idx)

        use_canon = self.coord_frame_actual == "canon"
        align_atoms = self.align_atoms and (self.f_canon is not None or order_idx is not None)
        if align_atoms:
            atomic_numbers = self.z_canon[idx] if self.z_canon is not None else self._reorder_by_index(self.z[idx], order_idx)
            if use_canon and self.f_canon is not None:
                frac_coords = self.f_canon[idx]
            else:
                frac_coords = self._reorder_by_index(self.f[idx], order_idx)
            if self.atom_mask_canon is not None:
                atom_mask = self.atom_mask_canon[idx]
            else:
                atom_mask = self._reorder_by_index(self.atom_mask[idx], order_idx)
        else:
            atomic_numbers = self.z[idx]
            frac_coords = self.f_canon[idx] if use_canon and self.f_canon is not None else self.f[idx]
            atom_mask = self.atom_mask[idx]

        gram6 = self.gram6_canon[idx] if use_canon and self.gram6_canon is not None else self.gram6[idx]
        sample = {
            "atomic_numbers": atomic_numbers,
            "frac_coords": frac_coords,
            "atom_mask": atom_mask,
            "gram6": gram6,
        }
        lattice = None
        if use_canon and self.lattice_canon is not None:
            lattice = self.lattice_canon[idx]
        elif self.lattice is not None:
            lattice = self.lattice[idx]
        if lattice is not None:
            sample["lattice_matrix"] = lattice
        if self.z_canon is not None:
            sample["atomic_numbers_canon"] = self.z_canon[idx]
        if self.uvz is not None:
            uvz = self.uvz[idx]
            if not align_atoms and order_inv is not None:
                uvz = self._reorder_by_index(uvz, order_inv)
            sample["uvz"] = uvz
        if self.uv_angle is not None:
            uv_angle = self.uv_angle[idx]
            if not align_atoms and order_inv is not None:
                uv_angle = self._reorder_by_index(uv_angle, order_inv)
            sample["uv_angle"] = uv_angle
        if self.u is not None:
            u = self.u[idx]
            if not align_atoms and order_inv is not None:
                u = self._reorder_by_index(u, order_inv)
            sample["u"] = u
        if self.v is not None:
            v = self.v[idx]
            if not align_atoms and order_inv is not None:
                v = self._reorder_by_index(v, order_inv)
            sample["v"] = v
        if self.z_norm is not None:
            z_norm = self.z_norm[idx]
            if not align_atoms and order_inv is not None:
                z_norm = self._reorder_by_index(z_norm, order_inv)
            sample["z_norm"] = z_norm
        if self.t is not None:
            sample["t"] = self.t[idx]
        if self.a_hat is not None:
            sample["a_hat"] = self.a_hat[idx]
        if self.b_hat is not None:
            sample["b_hat"] = self.b_hat[idx]
        if self.n is not None:
            sample["n"] = self.n[idx]
        if self.lattice_param is not None:
            sample["lattice_param"] = self.lattice_param[idx]
        if self.counts_vector is not None:
            sample["counts_vector"] = self.counts_vector[idx]
        if self.order_idx is not None:
            sample["order_idx"] = self.order_idx[idx]
        if self.extra:
            for key, value in self.extra.items():
                sample[key] = value[idx]
        return sample

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def _lattice_to_gram6(lattice: torch.Tensor) -> torch.Tensor:
    """Convert lattice (3x3) to Gram 6D vector."""
    # Convention: lattice basis vectors are stored in rows (cart = frac @ lattice).
    gram = lattice.matmul(lattice.t())
    return torch.stack(
        [gram[0, 0], gram[1, 1], gram[2, 2], gram[0, 1], gram[0, 2], gram[1, 2]],
        dim=0,
    )


def _counts_vector(
    atomic_numbers: torch.Tensor, atom_mask: torch.Tensor, max_atomic_number: int = 118
) -> torch.Tensor:
    counts = torch.zeros(max_atomic_number, dtype=torch.long, device=atomic_numbers.device)
    valid = atom_mask > 0.5
    z = atomic_numbers[valid].long()
    keep = (z > 0) & (z <= max_atomic_number)
    if keep.any():
        z = z[keep] - 1
        counts.index_put_((z,), torch.ones_like(z, dtype=counts.dtype), accumulate=True)
    return counts
