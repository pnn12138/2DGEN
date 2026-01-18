from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot formation energy distributions.")
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.per_sample)
    values = [
        r.get("formation_energy_per_atom")
        for r in rows
        if r.get("formation_energy_per_atom") is not None
    ]
    pass_flags = [int(bool(r.get("formation_pass"))) for r in rows if "formation_pass" in r]
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if arr.size:
        plt.figure(figsize=(5, 3.6))
        plt.hist(arr, bins=args.bins, color="#3b6fb6", alpha=0.8)
        plt.title("formation_energy_per_atom")
        plt.tight_layout()
        plt.savefig(args.out_dir / "formation_energy_hist.png", dpi=150)
        plt.close()

    if pass_flags:
        plt.figure(figsize=(4, 3))
        rate = float(np.mean(pass_flags))
        plt.bar(["pass", "fail"], [rate, 1.0 - rate], color=["#4e9a51", "#d1495b"])
        plt.ylim(0, 1)
        plt.title("formation_pass_rate")
        plt.tight_layout()
        plt.savefig(args.out_dir / "formation_pass_rate.png", dpi=150)
        plt.close()

    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
