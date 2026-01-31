from __future__ import annotations

from typing import Tuple

import torch


def choose_vacuum_axis_torch(
    lattice: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.linalg.norm(lattice, dim=-1)
    invalid = (~torch.isfinite(lengths)) | (lengths <= 0)
    invalid_any = invalid.any(dim=-1)
    c_idx = torch.argmax(lengths, dim=-1)
    if invalid_any.any():
        c_idx = torch.where(invalid_any, torch.full_like(c_idx, 2), c_idx)
    c_len = lengths.gather(-1, c_idx.unsqueeze(-1)).squeeze(-1)
    c_len = torch.where(invalid_any, torch.full_like(c_len, float("nan")), c_len)
    return c_idx, c_len, lengths


__all__ = ["choose_vacuum_axis_torch"]
