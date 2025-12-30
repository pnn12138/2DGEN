from __future__ import annotations

import sys
from pathlib import Path

# Ensure the 2DGEN package directory is importable even though the folder starts with a digit.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from evaluate.plot_eval import main


if __name__ == "__main__":
    main()
