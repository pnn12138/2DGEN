"""Data utilities for 2D material prediction tasks.

Avoid importing heavy/optional deps (e.g., jarvis-tools) unless the Jdft2d
helpers are explicitly requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .jdft2d import DEFAULT_JDFT2D_DATASET, download_jdft2d_dataset

__all__ = ["download_jdft2d_dataset", "DEFAULT_JDFT2D_DATASET"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module("p_task.data.jdft2d")
    except ImportError as exc:  # pragma: no cover - only hit when optional deps missing
        raise ImportError(
            "jarvis-tools is required for Jdft2d helpers; install deps or avoid importing them."
        ) from exc
    return getattr(module, name)
