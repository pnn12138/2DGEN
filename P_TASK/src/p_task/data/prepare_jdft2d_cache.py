"""Download/cache the matbench_jdft2d dataset and build lightweight artifacts."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Tuple

import math
import pandas as pd
from pymatgen.core import Structure

DEFAULT_URL = "https://ml.materialsproject.org/projects/matbench_jdft2d.json.gz"
DEFAULT_CACHE_DIR = Path("/home/pnn/2dgen/P_TASK/data/jdft2d_cache")


def download_raw(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    with urllib.request.urlopen(url) as resp:  # noqa: S310 - trusted source
        dest.write_bytes(resp.read())


def load_raw_dataframe(raw_path: Path) -> pd.DataFrame:
    with gzip.open(raw_path, "rt") as f:
        obj = json.load(f)
    df = pd.DataFrame(obj["data"], columns=obj["columns"], index=obj["index"])
    return df


def build_cache(cache_dir: Path, *, force: bool = False) -> Tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "matbench_jdft2d.json.gz"
    meta_path = cache_dir / "jdft2d_meta.csv"
    struct_path = cache_dir / "structures.pkl"

    if meta_path.exists() and struct_path.exists() and not force:
        return meta_path, struct_path

    if not raw_path.exists() or force:
        download_raw(DEFAULT_URL, raw_path)

    df = load_raw_dataframe(raw_path)
    structures = [Structure.from_dict(s) for s in df["structure"]]
    n_digits = math.floor(math.log10(len(df))) + 1
    ids = [f"mb-jdft2d-{i+1:0{n_digits}d}" for i in range(len(df))]
    meta = pd.DataFrame(
        {
            "sample_id": ids,
            "formula": [s.composition.reduced_formula for s in structures],
            "exfoliation_en": df["exfoliation_en"].astype(float),
        }
    )

    meta.to_csv(meta_path, index=False)
    with open(struct_path, "wb") as f:
        pickle.dump(structures, f)

    return meta_path, struct_path


def main() -> None:  # pragma: no cover - CLI utility
    meta_path, struct_path = build_cache(DEFAULT_CACHE_DIR)
    print("Cached metadata:", meta_path)
    print("Cached structures:", struct_path)


if __name__ == "__main__":
    main()
