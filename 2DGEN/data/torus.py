from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch

# Default harmonic frequencies used for torus encoding of fractional coordinates.
DEFAULT_TORUS_FREQS: Tuple[int, ...] = (1, 2, 4, 8)


def torus_feature_dim(freqs: Sequence[int]) -> int:
    """Return flattened feature dimension for torus-encoded frac coords."""
    return 3 * 2 * len(freqs)


def wrap01_tensor(x: torch.Tensor) -> torch.Tensor:
    """Map values to [0, 1) on a per-element basis."""
    return x - torch.floor(x)


def wrap01_array(x: np.ndarray) -> np.ndarray:
    """Map numpy arrays to [0, 1) on a per-element basis."""
    return x - np.floor(x)


def _as_freq_tensor(freqs: Sequence[int], device, dtype) -> torch.Tensor:
    freq_tensor = torch.as_tensor(freqs, device=device, dtype=dtype)
    if freq_tensor.ndim != 1:
        raise ValueError("freqs must be a 1D sequence of integers.")
    return freq_tensor


def torus_encode(frac: torch.Tensor, freqs: Sequence[int] = DEFAULT_TORUS_FREQS) -> torch.Tensor:
    """
    Encode fractional coordinates on a 3-torus using sin/cos harmonics.

    Args:
        frac: tensor shaped (..., 3) with fractional coordinates.
        freqs: harmonic frequencies to encode.
    Returns:
        Tensor shaped (..., 3 * 2 * F) where F=len(freqs), ordered as [sin, cos].
    """
    frac = wrap01_tensor(frac)
    f = _as_freq_tensor(freqs, device=frac.device, dtype=frac.dtype)
    ang = 2 * math.pi * frac.unsqueeze(-1) * f  # (..., 3, F)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (..., 3, 2F)
    return emb.reshape(*frac.shape[:-1], torus_feature_dim(freqs))


def torus_encode_np(frac: np.ndarray, freqs: Sequence[int] = DEFAULT_TORUS_FREQS) -> np.ndarray:
    """
    Numpy counterpart of torus_encode for preprocessing.
    """
    frac = wrap01_array(frac)
    f = np.asarray(freqs, dtype=frac.dtype)
    if f.ndim != 1:
        raise ValueError("freqs must be a 1D sequence of integers.")
    ang = 2 * math.pi * frac[..., None] * f  # (..., 3, F)
    emb = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)  # (..., 3, 2F)
    return emb.reshape(*frac.shape[:-1], torus_feature_dim(freqs)).astype(np.float32)


def torus_decode(encoded: torch.Tensor, freqs: Sequence[int] = DEFAULT_TORUS_FREQS) -> torch.Tensor:
    """
    Recover fractional coordinates in [0,1) from torus-encoded features.

    Uses the fundamental frequency (first sin/cos pair); higher harmonics help
    training but are redundant for decoding.
    """
    feature_dim = torus_feature_dim(freqs)
    if encoded.shape[-1] != feature_dim:
        raise ValueError(f"Expected last dimension {feature_dim}, got {encoded.shape[-1]}.")
    f_len = len(freqs)
    encoded = encoded.reshape(*encoded.shape[:-1], 3, 2 * f_len)
    sin_part = encoded[..., :f_len]
    cos_part = encoded[..., f_len:]
    angle = torch.atan2(sin_part[..., 0], cos_part[..., 0])  # (..., 3)
    frac = angle / (2 * math.pi)
    return wrap01_tensor(frac)


def torus_decode_np(encoded: np.ndarray, freqs: Sequence[int] = DEFAULT_TORUS_FREQS) -> np.ndarray:
    """
    Numpy counterpart of torus_decode.
    """
    feature_dim = torus_feature_dim(freqs)
    if encoded.shape[-1] != feature_dim:
        raise ValueError(f"Expected last dimension {feature_dim}, got {encoded.shape[-1]}.")
    f_len = len(freqs)
    encoded = encoded.reshape(*encoded.shape[:-1], 3, 2 * f_len)
    sin_part = encoded[..., :f_len]
    cos_part = encoded[..., f_len:]
    angle = np.arctan2(sin_part[..., 0], cos_part[..., 0])
    frac = angle / (2 * math.pi)
    return wrap01_array(frac).astype(np.float32)


__all__ = [
    "DEFAULT_TORUS_FREQS",
    "torus_decode",
    "torus_decode_np",
    "torus_encode",
    "torus_encode_np",
    "torus_feature_dim",
    "wrap01_array",
    "wrap01_tensor",
]
