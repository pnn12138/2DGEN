#!/usr/bin/env python
"""Hydra training entry kept under scripts/ for matbench_jdft2d."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project src is importable when running from anywhere.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p_task.train import main  # pylint: disable=wrong-import-position


if __name__ == "__main__":
    main()
