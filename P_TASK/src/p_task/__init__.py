"""Core package for 2D material prediction tasks."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("p-task")
except PackageNotFoundError:  # pragma: no cover - during local dev installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
