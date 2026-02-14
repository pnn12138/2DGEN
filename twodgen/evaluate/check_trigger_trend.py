from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

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


def _finite_series(rows: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            values.append(v)
    return values


def _trend(values: List[float]) -> Dict[str, Any]:
    if len(values) < 2:
        return {
            "available": False,
            "reason": "insufficient_points",
            "first_half_mean": None,
            "second_half_mean": None,
            "delta_second_minus_first": None,
            "improved": None,
        }
    split = max(1, len(values) // 2)
    first = float(np.mean(np.asarray(values[:split], dtype=float)))
    second = float(np.mean(np.asarray(values[split:], dtype=float)))
    delta = second - first
    return {
        "available": True,
        "first_half_mean": first,
        "second_half_mean": second,
        "delta_second_minus_first": delta,
        "improved": bool(second < first),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether trigger proxies trend down in second half of training."
    )
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument(
        "--keys",
        type=str,
        default=(
            "post_project_trigger_rate_train_proxy,"
            "cond_violation_rate_train_proxy,"
            "vacuum_violation_rate_train_proxy"
        ),
        help="Comma-separated keys to evaluate.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.metrics_jsonl)
    keys = [k.strip() for k in str(args.keys).split(",") if k.strip()]
    if not rows:
        raise ValueError(f"No rows found in {args.metrics_jsonl}")
    if not keys:
        raise ValueError("--keys is empty")

    trends: Dict[str, Any] = {}
    passes: List[bool] = []
    for key in keys:
        values = _finite_series(rows, key)
        info = _trend(values)
        info["points"] = len(values)
        trends[key] = info
        if info.get("available"):
            passes.append(bool(info.get("improved")))

    passed = bool(passes) and all(passes)
    report = {
        "metrics_jsonl": str(args.metrics_jsonl),
        "rows": len(rows),
        "keys": keys,
        "trend": trends,
        "pass": passed,
    }
    out_path = args.out or (args.metrics_jsonl.parent / "trigger_trend_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved trigger trend report to {out_path}")
    print(f"pass={passed}")


if __name__ == "__main__":
    main()

