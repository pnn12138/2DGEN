from __future__ import annotations

from typing import Tuple

import numpy as np


def wrap01_array(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def pad_1d(values: np.ndarray, max_len: int, pad_value: float) -> Tuple[np.ndarray, np.ndarray]:
    trimmed = values[:max_len]
    padded = np.full((max_len,), pad_value, dtype=trimmed.dtype)
    padded[: len(trimmed)] = trimmed
    mask = np.zeros((max_len,), dtype=np.float32)
    mask[: len(trimmed)] = 1.0
    return padded, mask


def pad_2d(values: np.ndarray, max_len: int, pad_value: float) -> Tuple[np.ndarray, np.ndarray]:
    trimmed = values[:max_len]
    padded = np.full((max_len, 3), pad_value, dtype=np.float32)
    padded[: len(trimmed)] = trimmed.astype(np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    mask[: len(trimmed)] = 1.0
    return padded, mask


def pad_2d4(values: np.ndarray, max_len: int, pad_value: float) -> Tuple[np.ndarray, np.ndarray]:
    trimmed = values[:max_len]
    padded = np.full((max_len, 4), pad_value, dtype=np.float32)
    padded[: len(trimmed)] = trimmed.astype(np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    mask[: len(trimmed)] = 1.0
    return padded, mask


__all__ = ["pad_1d", "pad_2d", "pad_2d4", "wrap01_array"]
