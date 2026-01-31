from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _counts_vector(z: np.ndarray, atom_mask: np.ndarray, num_elements: int = 118) -> np.ndarray:
    counts = np.zeros((z.shape[0], num_elements), dtype=np.int64)
    valid = (atom_mask > 0.5) & (z > 0)
    batch_idx, atom_idx = np.where(valid)
    if batch_idx.size == 0:
        return counts
    elem_idx = z[batch_idx, atom_idx].astype(np.int64) - 1
    keep = (elem_idx >= 0) & (elem_idx < num_elements)
    if keep.any():
        np.add.at(counts, (batch_idx[keep], elem_idx[keep]), 1)
    return counts


def _lattice_to_gram6(lattice: np.ndarray) -> np.ndarray:
    gram = np.matmul(lattice, np.transpose(lattice, (0, 2, 1)))
    return np.stack(
        [
            gram[:, 0, 0],
            gram[:, 1, 1],
            gram[:, 2, 2],
            gram[:, 0, 1],
            gram[:, 0, 2],
            gram[:, 1, 2],
        ],
        axis=-1,
    ).astype(np.float32)


def _load_success_indices(per_sample_path: Path) -> List[int]:
    indices: List[int] = []
    with per_sample_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("success"):
                indices.append(int(row["id"]))
    return indices


def _filter_samples(samples: Dict[str, np.ndarray], indices: List[int]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v)[indices] for k, v in samples.items() if np.asarray(v).shape[0] >= max(indices, default=-1) + 1}


def _merge_npz(base: Dict[str, np.ndarray], extra: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    merged = dict(base)
    for key, val in extra.items():
        if key in merged and merged[key].shape[0] == base["z"].shape[0]:
            merged[key] = np.concatenate([merged[key], val], axis=0)
        elif key not in merged:
            merged[key] = val
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-train loop: sample -> eval -> append dataset.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sample-args", type=str, default="")
    parser.add_argument("--eval-args", type=str, default="")
    parser.add_argument("--base-npz", type=Path, default=None)
    parser.add_argument("--max-atoms", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / "samples"
    eval_dir = out_dir / "eval"

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
        "--max-atoms",
        str(args.max_atoms),
    ]
    if args.sample_args:
        sample_cmd += args.sample_args.split()
    _run(sample_cmd)

    eval_cmd = [
        "python",
        "-m",
        "twodgen.evaluate.eval_samples",
        "--samples",
        str(sample_dir / "samples.npz"),
        "--out-dir",
        str(eval_dir),
    ]
    if args.eval_args:
        eval_cmd += args.eval_args.split()
    _run(eval_cmd)

    success_indices = _load_success_indices(eval_dir / "per_sample.jsonl")
    samples = dict(np.load(sample_dir / "samples.npz"))
    if not success_indices:
        summary = {"selected": 0, "total": int(samples["z"].shape[0])}
        (out_dir / "self_train_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print("No successful samples found; nothing to append.")
        return

    filtered = _filter_samples(samples, success_indices)
    z = filtered["z"]
    atom_mask = filtered["atom_mask"]
    lattice = filtered["lattice"]
    gram6 = _lattice_to_gram6(lattice)
    counts_vec = _counts_vector(z, atom_mask)
    new_npz = {
        "z": z,
        "f": filtered["frac"],
        "atom_mask": atom_mask,
        "lattice": lattice,
        "gram6": gram6,
        "counts_vector": counts_vec,
        "max_atoms": np.array(int(args.max_atoms)),
        "g_scale": np.array(1.0, dtype=np.float32),
        "gram6_convention": np.array("row_lattice"),
        "schema_version": np.array("self_train_v1"),
        "coord_frame": np.array("raw"),
    }

    if args.base_npz is not None:
        base = dict(np.load(args.base_npz))
        merged = _merge_npz(base, new_npz)
        out_path = out_dir / "self_train_merged.npz"
        np.savez_compressed(out_path, **merged)
    else:
        out_path = out_dir / "self_train.npz"
        np.savez_compressed(out_path, **new_npz)

    summary = {
        "selected": len(success_indices),
        "total": int(samples["z"].shape[0]),
        "output": str(out_path),
        "success_indices": success_indices,
    }
    (out_dir / "self_train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(f"Saved self-train dataset to {out_path}")


if __name__ == "__main__":
    main()
