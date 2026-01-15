from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _try_git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _try_git_status_dirty(cwd: Optional[Path] = None) -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def collect_run_metadata(argv: Optional[list[str]] = None) -> Dict[str, Any]:
    """
    Collect lightweight metadata for reproducibility.

    This function is intentionally best-effort: git info may be missing in exported
    environments, and we don't want to fail a run because of that.
    """
    cwd = Path.cwd()
    metadata: Dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "argv": list(argv) if argv is not None else list(sys.argv),
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    metadata["git"] = {
        "commit": _try_git_commit(cwd=cwd),
        "dirty": _try_git_status_dirty(cwd=cwd),
    }
    try:
        import numpy as np

        metadata["numpy"] = np.__version__
    except Exception:
        metadata["numpy"] = None
    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        metadata["torch"] = None
        metadata["cuda_available"] = None
    return metadata

