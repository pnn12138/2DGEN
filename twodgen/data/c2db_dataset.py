from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


class C2DBTokenNPZDataset(Dataset):
    """
    Dataset that reads token caches saved by prepare_c2db_tokens.py.
    """

    def __init__(self, npz_path: Path | str, align_atoms: bool = True, coord_frame: str = "raw") -> None:
        super().__init__()
        data = np.load(npz_path)
        deprecated_keys = {"nbr_idx", "nbr_dist", "nbr_mask"}
        found_deprecated = deprecated_keys.intersection(set(data.files))
        if found_deprecated:
            raise ValueError(
                "Legacy neighbor caches are no longer supported in token npz files. "
                f"Please regenerate the cache without {sorted(found_deprecated)}."
            )
        gram6_convention = data["gram6_convention"].item() if "gram6_convention" in data else None
        if gram6_convention is None:
            raise ValueError(
                "Legacy token cache detected (missing `gram6_convention`). "
                "This repo now assumes row-vector lattices (cart = frac @ lattice) and "
                "Gram6 computed from `G = lattice @ lattice^T`. "
                "Please re-run preprocessing to regenerate token caches."
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
        self.min_dist = torch.from_numpy(data["min_dist"]).float() if "min_dist" in data else None
        self.collision_risk = (
            torch.from_numpy(data["collision_risk"]).long() if "collision_risk" in data else None
        )
        self.min_dist_cut = float(np.asarray(data["min_dist_cut"]).reshape(-1)[0]) if "min_dist_cut" in data else None
        self.min_dist_pbc_mask = None
        if "min_dist_pbc_mask" in data:
            mask_arr = np.asarray(data["min_dist_pbc_mask"]).astype(int).reshape(-1).tolist()
            if len(mask_arr) == 3:
                self.min_dist_pbc_mask = tuple(int(x) for x in mask_arr)
        self.schema_version = data["schema_version"].item() if "schema_version" in data else None
        self.coord_frame = data["coord_frame"].item() if "coord_frame" in data else None
        self.align_atoms = align_atoms
        self.coord_frame_requested = coord_frame
        self.coord_frame_actual = coord_frame
        if coord_frame == "canon" and (self.f_canon is None or self.gram6_canon is None):
            self.coord_frame_actual = "raw"
            warnings.warn(
                "coord_frame=canon requested but npz lacks canonical fields; falling back to raw.",
                RuntimeWarning,
                stacklevel=2,
            )
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
            "min_dist",
            "collision_risk",
            "min_dist_cut",
            "min_dist_pbc_mask",
            "preprocess_v3",
            "preprocess_version",
            "eps_area",
            "eps_inv",
            "round_prec",
            "z_norm_clip",
            "neighbor_k",
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
        if self.min_dist is not None:
            sample["min_dist"] = self.min_dist[idx]
        if self.collision_risk is not None:
            sample["collision_risk"] = self.collision_risk[idx]
        if self.extra:
            for key, value in self.extra.items():
                sample[key] = value[idx]
        return sample

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
