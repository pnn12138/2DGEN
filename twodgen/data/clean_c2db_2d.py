from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure


def _wrap01_array(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _thickness_vacuum(frac_1d: np.ndarray, c_len: float) -> Tuple[float, float]:
    if frac_1d.size == 0:
        return float("nan"), float("nan")
    coords = np.sort(_wrap01_array(frac_1d.astype(float)))
    if coords.size == 1:
        thickness = 0.0
        return thickness, c_len - thickness
    gaps = np.diff(coords, axis=0).flatten().tolist()
    gaps.append(1.0 - (coords[-1] - coords[0]))
    max_gap = max(gaps)
    thickness = (1.0 - max_gap) * c_len
    vacuum = c_len - thickness
    return float(thickness), float(vacuum)


def _mic_dist_and_shifts(
    frac: np.ndarray, lattice: np.ndarray, pbc_mask: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    df = frac[:, None, :] - frac[None, :, :]
    shifts_1d = (-1.0, 0.0, 1.0)
    zeros_1d = (0.0,)
    components = [
        shifts_1d if pbc_mask[0] == 1 else zeros_1d,
        shifts_1d if pbc_mask[1] == 1 else zeros_1d,
        shifts_1d if pbc_mask[2] == 1 else zeros_1d,
    ]
    shifts_all = np.asarray(list(_cartesian_product(components)), dtype=float)  # (S, 3)
    df_shifted = df[:, :, None, :] - shifts_all[None, None, :, :]  # (N, N, S, 3)
    dr = df_shifted @ lattice  # (N, N, S, 3)
    dist_all = np.linalg.norm(dr, axis=-1)  # (N, N, S)
    best_idx = np.argmin(dist_all, axis=-1)  # (N, N)
    dist = np.take_along_axis(dist_all, best_idx[:, :, None], axis=-1)[:, :, 0]
    shifts = shifts_all[best_idx]
    np.fill_diagonal(dist, np.inf)
    return dist, shifts


def _cartesian_product(components: Sequence[Sequence[float]]) -> Iterable[Tuple[float, float, float]]:
    if len(components) != 3:
        raise ValueError("expected 3 components for cartesian product")
    for a in components[0]:
        for b in components[1]:
            for c in components[2]:
                yield (a, b, c)


def _choose_vacuum_axis(lattice: np.ndarray) -> Tuple[int, float, np.ndarray]:
    lengths = np.linalg.norm(lattice, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        return 2, float("nan"), lengths
    c_idx = int(np.argmax(lengths))
    return c_idx, float(lengths[c_idx]), lengths


def _source_bucket_from_row(row: Dict[str, Any]) -> str:
    candidates = [
        "exp/theo",
        "exp_theo",
        "exp_theory",
        "source",
        "source_type",
    ]
    raw = None
    for key in candidates:
        if key in row and row[key] is not None and str(row[key]).strip():
            raw = str(row[key]).strip().lower()
            break
    if raw is None:
        return "unknown"
    if "exp" in raw or raw in {"e", "experimental"}:
        return "exp"
    if "theo" in raw or "dft" in raw or raw in {"t", "theoretical", "sim"}:
        return "theo"
    return "unknown"


@dataclass(frozen=True)
class C2DB2DQualityConfig:
    max_atoms: int
    min_vacuum: float
    bond_cut: float
    collision_risk_cut: float
    vacuum_risk_margin: float


def analyze_cif(cif_str: str, cfg: C2DB2DQualityConfig) -> Dict[str, Any]:
    structure = Structure.from_str(cif_str, fmt="cif")
    n_atoms = int(len(structure))
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    frac = _wrap01_array(np.asarray(structure.frac_coords, dtype=float))

    c_idx, c_len, lengths = _choose_vacuum_axis(lattice)
    pbc_mask_slab = (1, 1, 1)
    pbc_mask_slab = tuple(0 if i == c_idx else 1 for i in range(3))  # type: ignore[assignment]

    thickness, vacuum = _thickness_vacuum(frac[:, c_idx] if n_atoms else np.zeros((0,)), c_len)

    min_dist_slab = float("inf")
    cross_vacuum_bond = False
    if n_atoms > 1 and np.all(np.isfinite(lattice)):
        dist_slab, _ = _mic_dist_and_shifts(frac, lattice, pbc_mask=pbc_mask_slab)  # type: ignore[arg-type]
        min_dist_slab = float(np.min(dist_slab)) if dist_slab.size else float("inf")

        dist_3d, shifts_3d = _mic_dist_and_shifts(frac, lattice, pbc_mask=(1, 1, 1))
        below = (dist_3d < cfg.bond_cut) & np.isfinite(dist_3d)
        if np.any(below):
            shifts_c = shifts_3d[..., c_idx]
            cross_vacuum_bond = bool(np.any(below & (shifts_c != 0.0)))

    hard_fail_reasons: List[str] = []
    if n_atoms > cfg.max_atoms:
        hard_fail_reasons.append("too_many_atoms")
    if not (math.isfinite(vacuum) and vacuum >= cfg.min_vacuum):
        hard_fail_reasons.append("low_vacuum")
    if cross_vacuum_bond:
        hard_fail_reasons.append("cross_vacuum_bond")

    quality_tags: List[str] = []
    if math.isfinite(min_dist_slab) and min_dist_slab < cfg.collision_risk_cut:
        quality_tags.append("collision-risk")
    if cross_vacuum_bond:
        quality_tags.append("cross-vacuum-risk")
    if math.isfinite(vacuum) and vacuum < cfg.min_vacuum + cfg.vacuum_risk_margin:
        quality_tags.append("low-vacuum-risk")

    hard_pass = len(hard_fail_reasons) == 0
    quality_bucket = "good" if hard_pass and not quality_tags else "risk" if hard_pass else "bad"
    return {
        "n_atoms": n_atoms,
        "vacuum_axis_len": float(c_len),
        "thickness": thickness,
        "vacuum": vacuum,
        "min_dist_slab": min_dist_slab,
        "cross_vacuum_bond": bool(cross_vacuum_bond),
        "hard_pass": bool(hard_pass),
        "hard_fail_reason": "+".join(hard_fail_reasons),
        "quality_tags": ",".join(quality_tags),
        "quality_bucket": quality_bucket,
    }


def _iter_rows(csv_path: Path, chunksize: int, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    read_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for row in chunk.to_dict(orient="records"):
            yield row
            read_rows += 1
            if limit is not None and read_rows >= limit:
                return


def _write_csv(path: Path, fieldnames: Sequence[str]) -> Tuple[Any, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=list(fieldnames))
    writer.writeheader()
    return f, writer


def run_cleaning(
    csv_path: Path,
    out_dir: Path,
    cfg: C2DB2DQualityConfig,
    limit: Optional[int],
    chunksize: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = out_dir / "c2db_audit_2d.csv"
    clean_csv = out_dir / "c2db_clean_2d.csv"
    quality_jsonl = out_dir / "c2db_quality.jsonl"
    report_json = out_dir / "c2db_clean_report.json"

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    extra_cols = [
        "source_bucket",
        "n_atoms",
        "vacuum_axis_len",
        "thickness",
        "vacuum",
        "min_dist_slab",
        "cross_vacuum_bond",
        "hard_pass",
        "hard_fail_reason",
        "quality_tags",
        "quality_bucket",
    ]
    fieldnames = header + [c for c in extra_cols if c not in header]

    audit_f, audit_writer = _write_csv(audit_csv, fieldnames)
    clean_f, clean_writer = _write_csv(clean_csv, fieldnames)
    qf = quality_jsonl.open("w", encoding="utf-8")

    total = 0
    cleaned = 0
    parse_errors = 0
    fail_reason_counts: Counter[str] = Counter()
    fail_combo_counts: Counter[str] = Counter()
    tags_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    by_reason_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    kept_metrics: Dict[str, List[float]] = defaultdict(list)
    all_metrics: Dict[str, List[float]] = defaultdict(list)

    try:
        for row in _iter_rows(csv_path, chunksize=chunksize, limit=limit):
            total += 1
            material_id = str(row.get("material_id", ""))
            source_bucket = _source_bucket_from_row(row)
            source_counts[source_bucket] += 1

            cif = row.get("cif")
            analysis: Dict[str, Any]
            if not isinstance(cif, str) or not cif.strip():
                parse_errors += 1
                analysis = {
                    "n_atoms": 0,
                    "vacuum_axis_len": float("nan"),
                    "thickness": float("nan"),
                    "vacuum": float("nan"),
                    "min_dist_slab": float("nan"),
                    "cross_vacuum_bond": False,
                    "hard_pass": False,
                    "hard_fail_reason": "empty_cif",
                    "quality_tags": "parse-error",
                    "quality_bucket": "bad",
                }
            else:
                try:
                    analysis = analyze_cif(cif, cfg=cfg)
                except Exception:
                    parse_errors += 1
                    analysis = {
                        "n_atoms": 0,
                        "vacuum_axis_len": float("nan"),
                        "thickness": float("nan"),
                        "vacuum": float("nan"),
                        "min_dist_slab": float("nan"),
                        "cross_vacuum_bond": False,
                        "hard_pass": False,
                        "hard_fail_reason": "parse_error",
                        "quality_tags": "parse-error",
                        "quality_bucket": "bad",
                    }

            out_row = dict(row)
            out_row["source_bucket"] = source_bucket
            out_row.update(analysis)

            audit_writer.writerow(out_row)
            for k in ("n_atoms", "vacuum", "thickness", "min_dist_slab"):
                v = out_row.get(k)
                if isinstance(v, (int, float)):
                    all_metrics[k].append(float(v))

            tags = [t for t in str(out_row.get("quality_tags", "")).split(",") if t]
            for t in tags:
                tags_counts[t] += 1

            hard_pass = bool(out_row.get("hard_pass", False))
            if hard_pass:
                cleaned += 1
                clean_writer.writerow(out_row)
                for k in ("n_atoms", "vacuum", "thickness", "min_dist_slab"):
                    v = out_row.get(k)
                    if isinstance(v, (int, float)):
                        kept_metrics[k].append(float(v))
            else:
                combo = str(out_row.get("hard_fail_reason", "")).strip() or "unknown"
                fail_combo_counts[combo] += 1
                for reason in combo.split("+"):
                    if not reason:
                        continue
                    fail_reason_counts[reason] += 1
                    by_reason_metrics[reason]["n_atoms"].append(float(out_row.get("n_atoms", float("nan"))))
                    by_reason_metrics[reason]["vacuum"].append(float(out_row.get("vacuum", float("nan"))))
                    by_reason_metrics[reason]["thickness"].append(float(out_row.get("thickness", float("nan"))))
                    by_reason_metrics[reason]["min_dist_slab"].append(float(out_row.get("min_dist_slab", float("nan"))))

            q_payload = {
                "material_id": material_id,
                "source_bucket": source_bucket,
                "quality_bucket": out_row.get("quality_bucket"),
                "quality_tags": tags,
                "hard_pass": hard_pass,
                "hard_fail_reason": out_row.get("hard_fail_reason"),
                "n_atoms": out_row.get("n_atoms"),
                "vacuum": out_row.get("vacuum"),
                "thickness": out_row.get("thickness"),
                "min_dist_slab": out_row.get("min_dist_slab"),
                "cross_vacuum_bond": out_row.get("cross_vacuum_bond"),
            }
            qf.write(json.dumps(q_payload, ensure_ascii=True) + "\n")

    finally:
        audit_f.close()
        clean_f.close()
        qf.close()

    report: Dict[str, Any] = {
        "input_csv": str(csv_path),
        "config": {
            "max_atoms": cfg.max_atoms,
            "min_vacuum": cfg.min_vacuum,
            "bond_cut": cfg.bond_cut,
            "collision_risk_cut": cfg.collision_risk_cut,
            "vacuum_risk_margin": cfg.vacuum_risk_margin,
        },
        "counts": {
            "total": total,
            "cleaned": cleaned,
            "cleaned_ratio": (cleaned / total) if total else 0.0,
            "parse_errors": parse_errors,
            "parse_error_ratio": (parse_errors / total) if total else 0.0,
        },
        "hard_fail_reasons": dict(fail_reason_counts),
        "hard_fail_combinations": dict(fail_combo_counts.most_common(50)),
        "quality_tags": dict(tags_counts),
        "source_buckets": dict(source_counts),
        "metrics_all": {k: _summary_stats(v) for k, v in all_metrics.items()},
        "metrics_kept": {k: _summary_stats(v) for k, v in kept_metrics.items()},
        "metrics_by_fail_reason": {
            reason: {k: _summary_stats(v) for k, v in metrics.items()}
            for reason, metrics in by_reason_metrics.items()
        },
        "artifacts": {
            "audit_csv": str(audit_csv),
            "clean_csv": str(clean_csv),
            "quality_jsonl": str(quality_jsonl),
        },
    }
    report_json.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-filter C2DB for 2D slab quality + emit labels.")
    parser.add_argument("--csv", type=Path, default=Path("data/C2DB/c2db_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/C2DB/clean"))
    parser.add_argument("--max-atoms", type=int, default=24)
    parser.add_argument("--min-vacuum", type=float, default=15.0)
    parser.add_argument("--bond-cut", type=float, default=3.0)
    parser.add_argument("--collision-risk-cut", type=float, default=1.8)
    parser.add_argument("--vacuum-risk-margin", type=float, default=2.0)
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Issues encountered while parsing CIF: .*rounded to ideal values.*",
        category=UserWarning,
    )
    args = parse_args(argv)
    cfg = C2DB2DQualityConfig(
        max_atoms=int(args.max_atoms),
        min_vacuum=float(args.min_vacuum),
        bond_cut=float(args.bond_cut),
        collision_risk_cut=float(args.collision_risk_cut),
        vacuum_risk_margin=float(args.vacuum_risk_margin),
    )
    report = run_cleaning(
        csv_path=args.csv,
        out_dir=args.out_dir,
        cfg=cfg,
        limit=args.limit,
        chunksize=int(args.chunksize),
    )
    print(
        f"Cleaned {report['counts']['cleaned']}/{report['counts']['total']} "
        f"({report['counts']['cleaned_ratio']:.3f}) -> {Path(report['artifacts']['clean_csv'])}"
    )
    print(f"Report: {args.out_dir / 'c2db_clean_report.json'}")


if __name__ == "__main__":
    main()
