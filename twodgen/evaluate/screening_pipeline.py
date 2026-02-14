from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter

from twodgen.common.run_metadata import collect_run_metadata
from twodgen.evaluate.run_layout import (
    RUN_METADATA_SCHEMA_VERSION,
    atomic_write_json,
    config_hash,
    make_schema_payload,
)


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _energy_key(row: Dict[str, Any]) -> float:
    val = row.get("formation_energy_per_atom")
    if isinstance(val, (int, float)) and np.isfinite(float(val)):
        return float(val)
    val2 = row.get("energy_mlip")
    if isinstance(val2, (int, float)) and np.isfinite(float(val2)):
        return float(val2)
    return float("inf")


def _composition_feature(z_row: np.ndarray, mask_row: np.ndarray, dim: int = 118) -> np.ndarray:
    feat = np.zeros((dim,), dtype=float)
    valid = z_row[(mask_row > 0.5) & (z_row > 0)].astype(int)
    if valid.size == 0:
        return feat
    uniq, cnt = np.unique(valid, return_counts=True)
    keep = (uniq >= 1) & (uniq <= dim)
    if np.any(keep):
        feat[uniq[keep] - 1] = cnt[keep].astype(float)
    s = feat.sum()
    if s > 0:
        feat /= s
    return feat


def _sample_feature(samples: Dict[str, np.ndarray], sample_id: int, row: Dict[str, Any]) -> np.ndarray:
    z_row = np.asarray(samples["z"][sample_id])
    mask_row = np.asarray(samples["atom_mask"][sample_id])
    comp = _composition_feature(z_row, mask_row)
    n_atoms = float(row.get("n_atoms", np.sum(mask_row > 0.5)))
    inplane_area = float(row.get("inplane_area", 0.0))
    inplane_gamma = float(row.get("inplane_gamma", 90.0))
    tail = np.asarray(
        [
            n_atoms / 24.0,
            np.log1p(max(inplane_area, 0.0)),
            inplane_gamma / 180.0,
        ],
        dtype=float,
    )
    return np.concatenate([comp, tail], axis=0)


def _fps(indices: List[int], feat: np.ndarray, k: int) -> List[int]:
    if not indices or k <= 0:
        return []
    k = min(k, len(indices))
    chosen: List[int] = [indices[0]]
    chosen_mask = np.zeros((len(indices),), dtype=bool)
    chosen_mask[0] = True
    min_dist = np.linalg.norm(feat - feat[0][None, :], axis=1)
    for _ in range(1, k):
        candidate = int(np.argmax(np.where(chosen_mask, -1.0, min_dist)))
        if chosen_mask[candidate]:
            break
        chosen.append(indices[candidate])
        chosen_mask[candidate] = True
        d = np.linalg.norm(feat - feat[candidate][None, :], axis=1)
        min_dist = np.minimum(min_dist, d)
    return chosen


def _write_candidate_cif(samples: Dict[str, np.ndarray], sample_id: int, path: Path) -> None:
    z_row = np.asarray(samples["z"][sample_id])
    frac_row = np.asarray(samples["frac"][sample_id])
    lattice = np.asarray(samples["lattice"][sample_id])
    mask = np.asarray(samples["atom_mask"][sample_id]) > 0.5
    z_valid = z_row[mask].astype(int)
    frac_valid = frac_row[mask]
    species = [Element.from_Z(int(z)).symbol for z in z_valid.tolist()]
    structure = Structure(lattice=lattice, species=species, coords=frac_valid, coords_are_cartesian=False)
    writer = CifWriter(structure)
    writer.write_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLIP->DFT screening funnel pipeline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-m", type=int, default=500, help="Energy pre-filter size before diversity FPS.")
    parser.add_argument("--top-k", type=int, default=100, help="Final diverse candidate count.")
    parser.add_argument("--sample-args", type=str, default="")
    parser.add_argument("--experiment-id", type=str, default="E4_1")
    parser.add_argument("--protocol", type=str, default="final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    eval_dir = out_dir / "eval"
    candidates_dir = out_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    sample_cmd = [
        "python",
        "-m",
        "twodgen.scrip.sample_tokens",
        "--checkpoint",
        str(args.checkpoint),
        "--out-dir",
        str(sample_dir),
        "--num-samples",
        str(args.num_samples),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--eval",
        "--eval-out-dir",
        str(eval_dir),
        "--relax",
    ]
    if args.npz is not None:
        sample_cmd += ["--npz", str(args.npz)]
    if args.sample_args:
        sample_cmd += shlex.split(args.sample_args)
    _run(sample_cmd)

    per_sample_path = eval_dir / "per_sample.jsonl"
    if not per_sample_path.exists():
        raise FileNotFoundError(f"Missing per_sample.jsonl: {per_sample_path}")
    rows = _load_jsonl(per_sample_path)
    samples_npz = sample_dir / "samples.npz"
    samples = dict(np.load(samples_npz))

    total = len(rows)
    geom_rows = [r for r in rows if bool(r.get("success_geom"))]
    energy_rows = [r for r in geom_rows if bool(r.get("energy_available"))]
    ranked = sorted(energy_rows, key=_energy_key)
    top_m_rows = ranked[: max(0, int(args.top_m))]

    ids = [int(r["id"]) for r in top_m_rows]
    feat = np.stack([_sample_feature(samples, sid, row) for sid, row in zip(ids, top_m_rows)], axis=0) if ids else np.zeros((0, 121))
    selected_ids = set(_fps(ids, feat, int(args.top_k)))

    csv_path = out_dir / "screening.csv"
    fields = [
        "sample_id",
        "rank_energy",
        "selected_top_k",
        "formation_energy_per_atom",
        "energy_mlip",
        "success_geom",
        "energy_available",
        "run_path",
        "samples_path",
        "candidate_cif",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(top_m_rows, start=1):
            sid = int(row["id"])
            selected = sid in selected_ids
            cif_name = ""
            if selected:
                cif_name = f"candidate_{sid:05d}.cif"
                _write_candidate_cif(samples, sid, candidates_dir / cif_name)
            writer.writerow(
                {
                    "sample_id": sid,
                    "rank_energy": rank,
                    "selected_top_k": int(selected),
                    "formation_energy_per_atom": row.get("formation_energy_per_atom"),
                    "energy_mlip": row.get("energy_mlip"),
                    "success_geom": int(bool(row.get("success_geom"))),
                    "energy_available": int(bool(row.get("energy_available"))),
                    "run_path": str(out_dir),
                    "samples_path": str(samples_npz),
                    "candidate_cif": cif_name,
                }
            )

    summary = {
        "total_generated": int(total),
        "after_geom_gate": int(len(geom_rows)),
        "after_energy_gate": int(len(energy_rows)),
        "after_top_m": int(len(top_m_rows)),
        "after_top_k_diverse": int(len(selected_ids)),
        "top_m": int(args.top_m),
        "top_k": int(args.top_k),
        "selection_algorithm": {
            "name": "topM_energy_then_fps",
            "energy_metric": "formation_energy_per_atom_fallback_energy_mlip",
            "feature": "composition_118 + [n_atoms, log1p(inplane_area), inplane_gamma]",
        },
        "screening_csv": str(csv_path),
        "candidates_dir": str(candidates_dir),
    }
    (out_dir / "screening_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    cfg_hash = config_hash(
        {
            "checkpoint": str(args.checkpoint),
            "npz": str(args.npz) if args.npz is not None else None,
            "num_samples": int(args.num_samples),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "top_m": int(args.top_m),
            "top_k": int(args.top_k),
            "sample_args": str(args.sample_args),
        }
    )
    run_meta = make_schema_payload(
        schema_version=RUN_METADATA_SCHEMA_VERSION,
        payload={
            "run_metadata": collect_run_metadata(),
            "selection_algorithm": summary["selection_algorithm"],
            "funnel": {
                "total_generated": summary["total_generated"],
                "after_geom_gate": summary["after_geom_gate"],
                "after_energy_gate": summary["after_energy_gate"],
                "after_top_m": summary["after_top_m"],
                "after_top_k_diverse": summary["after_top_k_diverse"],
            },
        },
        experiment_id=str(args.experiment_id),
        seed=int(args.seed),
        protocol=str(args.protocol),
        config_hash_value=cfg_hash,
    )
    atomic_write_json(out_dir / "run_metadata.json", run_meta)
    print(f"Saved screening outputs to {out_dir}")


if __name__ == "__main__":
    main()
