from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _rate(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [int(bool(r.get(key))) for r in rows if key in r]
    return float(np.mean(vals)) if vals else 0.0


def _rate_any(rows: List[Dict[str, Any]], keys: tuple[str, ...]) -> float:
    for key in keys:
        vals = [int(bool(r.get(key))) for r in rows if key in r]
        if vals:
            return float(np.mean(vals))
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple evaluation scenarios.")
    parser.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="List of label:path_to_per_sample.jsonl",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for item in args.runs:
        if ":" not in item:
            raise ValueError("Each run must be in label:path format.")
        label, path_str = item.split(":", 1)
        path = Path(path_str)
        per_sample = _load_jsonl(path)
        rows.append(
            {
                "label": label,
                "path": str(path),
                "total": len(per_sample),
                "valid_rate": _rate(per_sample, "valid"),
                "cond_match_rate": _rate_any(per_sample, ("cond_exact_match", "cond_match")),
                "success_geom_rate": _rate_any(per_sample, ("success_geom",)),
                "success_energy_rate": _rate_any(per_sample, ("success_energy", "formation_pass")),
                "success_rate": _rate_any(per_sample, ("success",)),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scenarios": rows}, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved scenario comparison to {args.out}")


if __name__ == "__main__":
    main()
