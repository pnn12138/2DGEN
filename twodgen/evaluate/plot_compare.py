from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _parse_pbc_mask(value: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--pbc-mask must have three comma-separated values, e.g. 1,1,0")
    mask = tuple(int(p) for p in parts)
    if any(p not in (0, 1) for p in mask):
        raise ValueError("--pbc-mask values must be 0 or 1")
    return mask  # type: ignore[return-value]


def _min_dist_approx(frac: np.ndarray, lattice: np.ndarray, pbc_mask: Tuple[int, int, int]) -> float:
    df = frac[:, None, :] - frac[None, :, :]
    pbc = np.asarray(pbc_mask, dtype=float).reshape((1, 1, 3))
    df_mic = df - np.round(df) * pbc
    dr = df_mic @ lattice
    dist = np.linalg.norm(dr, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist)) if dist.size > 0 else float("inf")


def _min_dist_exact(frac: np.ndarray, lattice: np.ndarray, pbc_mask: Tuple[int, int, int]) -> float:
    df = frac[:, None, :] - frac[None, :, :]
    shifts_1d = (-1.0, 0.0, 1.0)
    zeros_1d = (0.0,)
    components = [
        shifts_1d if pbc_mask[0] == 1 else zeros_1d,
        shifts_1d if pbc_mask[1] == 1 else zeros_1d,
        shifts_1d if pbc_mask[2] == 1 else zeros_1d,
    ]
    shifts_all = np.asarray(list(itertools.product(*components)), dtype=float)  # (S, 3)
    df_shifted = df[:, :, None, :] - shifts_all[None, None, :, :]
    dr = df_shifted @ lattice
    dist_all = np.linalg.norm(dr, axis=-1)
    dist = np.min(dist_all, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist)) if dist.size > 0 else float("inf")


def _thickness_vacuum(frac: np.ndarray, c_len: float) -> Tuple[float, float]:
    if frac.size == 0:
        return float("nan"), float("nan")
    coords = np.sort(frac)
    gaps = np.diff(coords, axis=0).flatten().tolist()
    gaps.append(1.0 - (coords[-1] - coords[0]))
    max_gap = max(gaps)
    thickness = (1.0 - max_gap) * c_len
    vacuum = c_len - thickness
    return float(thickness), float(vacuum)


def _collect_metrics(
    samples: Dict[str, np.ndarray], pbc_mask: Tuple[int, int, int], mic_mode: str
) -> Dict[str, List[float]]:
    z = samples["z"]
    frac_key = "frac" if "frac" in samples else "f"
    frac = samples[frac_key]
    lattice = samples["lattice"]
    atom_mask = samples["atom_mask"]

    metrics = {
        "min_dist": [],
        "volume": [],
        "cond": [],
        "anisotropy": [],
        "thickness": [],
        "vacuum": [],
    }
    for i in range(z.shape[0]):
        mask = atom_mask[i] > 0.5
        frac_i = frac[i][mask]
        lattice_i = lattice[i]

        metrics["volume"].append(float(abs(np.linalg.det(lattice_i))))

        gram = lattice_i @ lattice_i.T
        eigvals = np.linalg.eigvalsh(gram)
        if np.any(eigvals <= 0.0) or not np.all(np.isfinite(eigvals)):
            cond = float("nan")
        else:
            cond = float(eigvals.max() / max(eigvals.min(), 1e-12))
        metrics["cond"].append(cond)

        lengths = np.linalg.norm(lattice_i, axis=1)
        c_idx = int(np.argmax(lengths))
        c_len = float(lengths[c_idx])
        ab = [l for j, l in enumerate(lengths) if j != c_idx]
        metrics["anisotropy"].append(float(c_len / max(np.mean(ab), 1e-8)))

        if frac_i.shape[0] > 0:
            if mic_mode == "approx":
                min_dist = _min_dist_approx(frac_i, lattice_i, pbc_mask=pbc_mask)
            else:
                min_dist = _min_dist_exact(frac_i, lattice_i, pbc_mask=pbc_mask)
            metrics["min_dist"].append(min_dist)
            t, v = _thickness_vacuum(frac_i[:, c_idx], c_len)
        else:
            metrics["min_dist"].append(float("nan"))
            t, v = float("nan"), float("nan")
        metrics["thickness"].append(t)
        metrics["vacuum"].append(v)

    return metrics


def _half_violin(ax, data: List[float], position: float, side: str, color: str) -> None:
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    parts = ax.violinplot(arr, positions=[position], widths=0.8, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.8)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        verts = body.get_paths()[0].vertices
        x = verts[:, 0]
        if side == "left":
            x[:] = np.minimum(x, position)
        else:
            x[:] = np.maximum(x, position)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sample vs dataset distributions.")
    parser.add_argument("--samples", type=Path, required=True, help="Path to samples.npz")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to dataset token npz")
    parser.add_argument("--out", type=Path, default=None, help="Output image path")
    parser.add_argument(
        "--coord-frame",
        type=str,
        default="raw",
        choices=["raw", "canon"],
        help="Coordinate frame to use for dataset frac coords.",
    )
    parser.add_argument(
        "--mic-mode",
        type=str,
        default="exact",
        choices=["exact", "approx"],
        help="Minimum image convention mode for min_dist.",
    )
    parser.add_argument(
        "--pbc-mask",
        type=str,
        default="1,1,0",
        help="Comma-separated PBC mask for MIC distance, e.g. 1,1,0 for slab.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples)
    dataset = np.load(args.dataset)
    pbc_mask = _parse_pbc_mask(args.pbc_mask)

    if args.coord_frame == "canon" and "f_canon" in dataset:
        dataset = dict(dataset)
        dataset["f"] = dataset["f_canon"]
        if "lattice_canon" in dataset:
            dataset["lattice"] = dataset["lattice_canon"]

    sample_metrics = _collect_metrics(samples, pbc_mask=pbc_mask, mic_mode=args.mic_mode)
    data_metrics = _collect_metrics(dataset, pbc_mask=pbc_mask, mic_mode=args.mic_mode)

    metrics = ["min_dist", "volume", "cond", "anisotropy", "thickness", "vacuum"]
    titles = {
        "min_dist": "min_dist",
        "volume": "volume",
        "cond": "cond",
        "anisotropy": "anisotropy",
        "thickness": "thickness",
        "vacuum": "vacuum",
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    for idx, key in enumerate(metrics):
        ax = axes[idx]
        _half_violin(ax, sample_metrics[key], position=1.0, side="left", color="#d1495b")
        _half_violin(ax, data_metrics[key], position=1.0, side="right", color="#3b6fb6")
        ax.set_title(titles[key])
        ax.set_xticks([1.0])
        ax.set_xticklabels(["sample | dataset"])
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for j in range(len(metrics), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    out_path = args.out
    if out_path is None:
        out_path = Path("outputs") / "eval_compare_violin.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
