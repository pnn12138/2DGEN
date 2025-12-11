#!/usr/bin/env python
"""Download/cache the JARVIS Jdft2d dataset artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p_task.data.jdft2d import (  # pylint: disable=wrong-import-position
    DEFAULT_CACHE_ROOT,
    DEFAULT_METADATA_FILENAME,
    DEFAULT_SPLIT_TEMPLATE,
    download_jdft2d_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Directory to place the Jdft2d CSV and splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when generating the train/val/test split.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download metadata even if cached files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = download_jdft2d_dataset(
        output_dir=args.output,
        split_seed=args.seed,
        split_template=DEFAULT_SPLIT_TEMPLATE,
        metadata_filename=DEFAULT_METADATA_FILENAME,
        force=args.force,
    )
    print(f"Metadata: {paths['metadata'].resolve()}")
    print(f"Split: {paths['split'].resolve()}")


if __name__ == "__main__":
    main()
