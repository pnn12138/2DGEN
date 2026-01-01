from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _safe_angles(lattice: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(lattice, axis=-1)
    angles = np.full((lattice.shape[0], 3), np.nan, dtype=np.float64)
    valid = np.all(lengths > 0, axis=1)
    if not np.any(valid):
        return angles
    mats = lattice[valid]
    a_vec = mats[:, 0]
    b_vec = mats[:, 1]
    c_vec = mats[:, 2]
    def ang(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        cos = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        cos = np.clip(cos, -1.0, 1.0)
        return np.degrees(np.arccos(cos))
    alpha = ang(b_vec, c_vec)
    beta = ang(a_vec, c_vec)
    gamma = ang(a_vec, b_vec)
    angles[valid] = np.stack([alpha, beta, gamma], axis=1)
    return angles


def _hist(ax: plt.Axes, data: np.ndarray, title: str, bins: int = 50) -> None:
    arr = data[np.isfinite(data)]
    if arr.size == 0:
        ax.set_title(f"{title} (empty)")
        return
    ax.hist(arr, bins=bins, color="#3b6fb6", alpha=0.8)
    ax.set_title(title)


def _mean_text(values: np.ndarray) -> str:
    arr = values[np.isfinite(values)]
    if arr.size == 0:
        return "nan"
    return f"{float(np.mean(arr)):.3f}"


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_npz(npz_path: Path, max_samples: int | None) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    payload = {key: data[key] for key in data.files}
    if max_samples is not None:
        for key, value in payload.items():
            if value.ndim >= 1 and value.shape[0] == payload["z"].shape[0]:
                payload[key] = value[:max_samples]
    return payload


def plot_lattice_distributions(lattice: np.ndarray, out_dir: Path) -> None:
    lengths = np.linalg.norm(lattice, axis=2)
    angles = _safe_angles(lattice)
    volume = np.abs(np.linalg.det(lattice))
    scube = np.power(volume, 1.0 / 3.0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _hist(axes[0], lengths[:, 0], f"a length (mean={_mean_text(lengths[:, 0])})")
    _hist(axes[1], lengths[:, 1], f"b length (mean={_mean_text(lengths[:, 1])})")
    _hist(axes[2], lengths[:, 2], f"c length (mean={_mean_text(lengths[:, 2])})")
    _save(fig, out_dir / "lattice_lengths.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _hist(axes[0], angles[:, 0], f"alpha (deg, mean={_mean_text(angles[:, 0])})")
    _hist(axes[1], angles[:, 1], f"beta (deg, mean={_mean_text(angles[:, 1])})")
    _hist(axes[2], angles[:, 2], f"gamma (deg, mean={_mean_text(angles[:, 2])})")
    _save(fig, out_dir / "lattice_angles.png")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    _hist(ax, volume, f"volume (mean={_mean_text(volume)})")
    _save(fig, out_dir / "lattice_volume.png")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    _hist(ax, scube, f"cell size (cuberoot volume, mean={_mean_text(scube)})")
    _save(fig, out_dir / "lattice_scube.png")


def plot_atom_distributions(z: np.ndarray, atom_mask: np.ndarray, out_dir: Path, top_k: int) -> None:
    counts = atom_mask.sum(axis=1)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    _hist(ax, counts, "atom count", bins=30)
    _save(fig, out_dir / "atom_counts.png")

    z_flat = z[atom_mask > 0.5].astype(int)
    z_flat = z_flat[z_flat > 0]
    if z_flat.size == 0:
        return
    max_z = int(z_flat.max())
    bincount = np.bincount(z_flat, minlength=max_z + 1)
    ids = np.argsort(bincount)[::-1][:top_k]
    ids = ids[bincount[ids] > 0]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([str(i) for i in ids], bincount[ids], color="#d1495b", alpha=0.85)
    ax.set_title(f"top-{len(ids)} elements by count (Z)")
    ax.set_xlabel("atomic number")
    ax.set_ylabel("count")
    _save(fig, out_dir / "element_topk.png")


def plot_frac_distributions(frac: np.ndarray, atom_mask: np.ndarray, out_dir: Path) -> None:
    mask = atom_mask > 0.5
    frac_valid = frac[mask]
    if frac_valid.size == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _hist(axes[0], frac_valid[:, 0], "frac x")
    _hist(axes[1], frac_valid[:, 1], "frac y")
    _hist(axes[2], frac_valid[:, 2], "frac z")
    _save(fig, out_dir / "frac_coords.png")


def plot_gram6_distributions(gram6: np.ndarray, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    labels = ["G11", "G22", "G33", "G12", "G13", "G23"]
    for idx, ax in enumerate(axes.flatten()):
        _hist(ax, gram6[:, idx], labels[idx])
    _save(fig, out_dir / "gram6.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot C2DB token dataset distributions.")
    parser.add_argument("--npz", type=Path, default=Path("data/C2DB/ache/c2db_tokens.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/C2DB/figures"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--top-elements", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.npz.exists():
        raise FileNotFoundError(f"Missing npz: {args.npz}")
    data = _load_npz(args.npz, args.max_samples)

    lattice = data.get("lattice")
    if lattice is None:
        raise ValueError("npz is missing lattice array")
    plot_lattice_distributions(lattice, args.out_dir)

    z = data.get("z")
    atom_mask = data.get("atom_mask")
    frac = data.get("f") if "f" in data else data.get("frac")
    if z is not None and atom_mask is not None:
        plot_atom_distributions(z, atom_mask, args.out_dir, args.top_elements)
    if frac is not None and atom_mask is not None:
        plot_frac_distributions(frac, atom_mask, args.out_dir)

    gram6 = data.get("gram6")
    if gram6 is not None:
        plot_gram6_distributions(gram6, args.out_dir)

    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
