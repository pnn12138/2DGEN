from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_TOTEN_RE = re.compile(r"TOTEN\s*=\s*([\-+0-9.eE]+)")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _parse_outcar_toten(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    energy = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _TOTEN_RE.search(line)
            if m:
                try:
                    energy = float(m.group(1))
                except Exception:
                    continue
    return energy


def _read_job_energy(job_dir: Path) -> Tuple[Optional[float], str]:
    result_json = job_dir / "result.json"
    if result_json.exists():
        try:
            data = json.loads(result_json.read_text(encoding="utf-8"))
            e = _float_or_none(data.get("final_energy", data.get("energy")))
            if e is not None:
                return e, "result_json"
        except Exception:
            pass
    energy_txt = job_dir / "energy.txt"
    if energy_txt.exists():
        text = energy_txt.read_text(encoding="utf-8").strip().split()
        for token in text:
            e = _float_or_none(token)
            if e is not None:
                return e, "energy_txt"
    outcar = job_dir / "OUTCAR"
    e_outcar = _parse_outcar_toten(outcar)
    if e_outcar is not None:
        return e_outcar, "OUTCAR"
    return None, "missing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import DFT energies and merge back into screening outputs.")
    parser.add_argument("--manifest", type=Path, required=True, help="dft_manifest.csv")
    parser.add_argument("--screening-csv", type=Path, required=True)
    parser.add_argument("--out-screening", type=Path, default=None)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = _read_csv(args.manifest)
    screening_rows = _read_csv(args.screening_csv)
    by_sample: Dict[int, Dict[str, Any]] = {}
    for row in manifest_rows:
        sample_id = int(float(row["sample_id"]))
        job_dir = Path(row["job_dir"])
        energy, source = _read_job_energy(job_dir)
        by_sample[sample_id] = {
            "job_id": row["job_id"],
            "job_dir": str(job_dir),
            "dft_energy": energy,
            "dft_source": source,
            "predicted_energy": _float_or_none(row.get("predicted_energy")),
        }

    updated_rows: List[Dict[str, Any]] = []
    available = 0
    deltas: List[float] = []
    for row in screening_rows:
        sid = int(float(row["sample_id"]))
        dft = by_sample.get(sid)
        row_out: Dict[str, Any] = dict(row)
        if dft is None:
            row_out["dft_job_id"] = ""
            row_out["dft_energy"] = ""
            row_out["dft_status"] = "not_exported"
            row_out["dft_source"] = ""
        elif dft["dft_energy"] is None:
            row_out["dft_job_id"] = dft["job_id"]
            row_out["dft_energy"] = ""
            row_out["dft_status"] = "missing_result"
            row_out["dft_source"] = dft["dft_source"]
        else:
            available += 1
            row_out["dft_job_id"] = dft["job_id"]
            row_out["dft_energy"] = float(dft["dft_energy"])
            row_out["dft_status"] = "ok"
            row_out["dft_source"] = dft["dft_source"]
            pred = dft["predicted_energy"]
            if pred is not None:
                deltas.append(float(dft["dft_energy"]) - float(pred))
        updated_rows.append(row_out)

    out_screening = args.out_screening or args.screening_csv.with_name("screening_with_dft.csv")
    out_screening.parent.mkdir(parents=True, exist_ok=True)
    if updated_rows:
        fieldnames = list(updated_rows[0].keys())
    else:
        fieldnames = []
    with out_screening.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in updated_rows:
                writer.writerow(row)

    summary = {
        "manifest": str(args.manifest),
        "screening_csv": str(args.screening_csv),
        "out_screening": str(out_screening),
        "total_rows": int(len(screening_rows)),
        "dft_available_rows": int(available),
        "dft_available_rate": float(available / max(len(screening_rows), 1)),
        "delta_energy_pred_vs_dft": {
            "count": int(len(deltas)),
            "mean": float(np.mean(deltas)) if deltas else None,
            "median": float(np.median(deltas)) if deltas else None,
            "q1": float(np.percentile(deltas, 25.0)) if deltas else None,
            "q3": float(np.percentile(deltas, 75.0)) if deltas else None,
        },
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved DFT import summary to {args.out_summary}")
    print(f"Saved merged screening CSV to {out_screening}")


if __name__ == "__main__":
    main()

