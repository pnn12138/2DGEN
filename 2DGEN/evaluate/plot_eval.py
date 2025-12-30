from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_per_sample(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _hist(values: List[float], title: str, out_path: Path, bins: int = 30) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    plt.figure(figsize=(4.8, 3.6))
    plt.hist(arr, bins=bins, color="#3b6fb6", alpha=0.8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _scatter(x: List[float], y: List[float], title: str, out_path: Path) -> None:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() == 0:
        return
    plt.figure(figsize=(4.8, 3.6))
    plt.scatter(x_arr[mask], y_arr[mask], s=10, alpha=0.7, color="#d1495b")
    plt.xlabel("thickness")
    plt.ylabel("vacuum")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot evaluation histograms and scatter plots.")
    parser.add_argument("--per-sample", type=Path, required=True, help="Path to per_sample.jsonl")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.per_sample.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_per_sample(args.per_sample)
    min_dist = [r.get("min_dist") for r in rows]
    volume = [r.get("volume") for r in rows]
    cond = [r.get("cond") for r in rows]
    thickness = [r.get("thickness") for r in rows]
    vacuum = [r.get("vacuum") for r in rows]
    anisotropy = [r.get("anisotropy") for r in rows]

    _hist(min_dist, "min_dist", out_dir / "min_dist_hist.png")
    _hist(volume, "volume", out_dir / "volume_hist.png")
    _hist(cond, "cond", out_dir / "cond_hist.png")
    _hist(anisotropy, "anisotropy", out_dir / "anisotropy_hist.png")
    _scatter(thickness, vacuum, "thickness_vs_vacuum", out_dir / "thickness_vacuum_scatter.png")

    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
