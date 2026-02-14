from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_npz_for_comp(path: Optional[Path]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if path is None:
        return None
    data = np.load(path)
    z = np.asarray(data["z"])
    atom_mask = np.asarray(data["atom_mask"])
    return z, atom_mask


def _parse_bins(text: str) -> np.ndarray:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) < 2:
        raise ValueError("Need at least two bin edges.")
    return np.asarray(vals, dtype=float)


def _comp_key(z_row: np.ndarray, mask_row: np.ndarray) -> str:
    valid = z_row[(mask_row > 0.5) & (z_row > 0)].astype(int)
    if valid.size == 0:
        return "empty"
    uniq, cnt = np.unique(valid, return_counts=True)
    parts = [f"{int(z)}:{int(c)}" for z, c in zip(uniq.tolist(), cnt.tolist())]
    return "|".join(parts)


def _coverage_from_bins(values: List[float], bins: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"occupied_bins": 0, "total_bins": int(max(len(bins) - 1, 0)), "coverage": 0.0}
    idx = np.digitize(arr, bins, right=False) - 1
    valid = idx[(idx >= 0) & (idx < len(bins) - 1)]
    occupied = len(np.unique(valid))
    total = int(len(bins) - 1)
    return {
        "occupied_bins": int(occupied),
        "total_bins": int(total),
        "coverage": float(occupied / max(total, 1)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute diversity coverage metrics.")
    parser.add_argument("--per-sample", type=Path, required=True, help="per_sample.jsonl from eval.")
    parser.add_argument("--samples", type=Path, default=None, help="samples.npz for composition coverage.")
    parser.add_argument("--train-npz", type=Path, default=None, help="Optional reference npz for relative coverage.")
    parser.add_argument("--bins-n-atoms", type=str, default="2,4,6,8,10,12,16,24")
    parser.add_argument("--bins-lattice", type=str, default="2,4,6,8,12,20,30,50")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.per_sample)
    n_atoms = [float(r.get("n_atoms", np.nan)) for r in rows]
    inplane_area = [float(r.get("inplane_area", np.nan)) for r in rows]
    sg = [int(r.get("spacegroup_number")) for r in rows if r.get("spacegroup_number") not in (None, "")]
    unique_sg = sorted(set(sg))

    bins_n_atoms = _parse_bins(args.bins_n_atoms)
    bins_lattice = _parse_bins(args.bins_lattice)
    cov_n_atoms = _coverage_from_bins(n_atoms, bins_n_atoms)
    cov_lattice = _coverage_from_bins(inplane_area, bins_lattice)

    comp_cov: Dict[str, Any] = {
        "available": False,
        "unique_generated": 0,
        "unique_reference": None,
        "relative_coverage": None,
    }
    sample_comp = _load_npz_for_comp(args.samples)
    if sample_comp is not None:
        z_s, m_s = sample_comp
        comp_gen = {_comp_key(z_s[i], m_s[i]) for i in range(z_s.shape[0])}
        comp_cov["available"] = True
        comp_cov["unique_generated"] = int(len(comp_gen))
        train_comp = _load_npz_for_comp(args.train_npz)
        if train_comp is not None:
            z_t, m_t = train_comp
            comp_ref = {_comp_key(z_t[i], m_t[i]) for i in range(z_t.shape[0])}
            comp_cov["unique_reference"] = int(len(comp_ref))
            comp_cov["relative_coverage"] = float(len(comp_gen) / max(len(comp_ref), 1))

    out = {
        "total_samples": int(len(rows)),
        "spacegroup": {
            "unique_count": int(len(unique_sg)),
            "coverage_vs_230": float(len(unique_sg) / 230.0),
        },
        "n_atoms_coverage": cov_n_atoms,
        "lattice_coverage": cov_lattice,
        "composition_coverage": comp_cov,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved diversity coverage to {args.out}")


if __name__ == "__main__":
    main()

