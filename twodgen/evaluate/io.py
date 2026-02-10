from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from twodgen.evaluate.run_layout import (
    FAILURE_BREAKDOWN_SCHEMA_VERSION,
    METRICS_SUMMARY_SCHEMA_VERSION,
)

_TIER0_NAMES = ("tier0_metrics.json", "tier0_metric.json", "tier0_metric.jsonl")
_TIER1_NAMES = ("tier1_2d_metrics.json", "tier1_metrics.json", "tier1_2d_metric.json")
_PER_SAMPLE_NAMES = ("per_sample.jsonl", "per_sanmple.jsonl")
_METRICS_SUMMARY_NAMES = ("metrics_summary.json",)
_FAILURE_BREAKDOWN_NAMES = ("failure_breakdown.json",)


def _first_existing(base: Path, names: tuple[str, ...]) -> Optional[Path]:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_schema_json(path: Path, expected_schema: str) -> Dict[str, Any]:
    data = _load_json(path)
    schema_version = data.get("schema_version")
    if schema_version is None:
        warnings.warn(
            f"{path.name} has no schema_version; treating as legacy format.",
            stacklevel=2,
        )
        return data
    if schema_version != expected_schema:
        warnings.warn(
            f"{path.name} schema_version={schema_version!r} does not match expected "
            f"{expected_schema!r}; parsing best-effort.",
            stacklevel=2,
        )
    return data


def load_eval_outputs(eval_dir: Path) -> Dict[str, Any]:
    eval_dir = Path(eval_dir)
    tier0_path = _first_existing(eval_dir, _TIER0_NAMES)
    tier1_path = _first_existing(eval_dir, _TIER1_NAMES)
    per_sample_path = _first_existing(eval_dir, _PER_SAMPLE_NAMES)
    metrics_summary_path = _first_existing(eval_dir, _METRICS_SUMMARY_NAMES)
    failure_breakdown_path = _first_existing(eval_dir, _FAILURE_BREAKDOWN_NAMES)

    if tier0_path is not None and tier0_path.name != _TIER0_NAMES[0]:
        warnings.warn(
            f"Using legacy tier0 filename {tier0_path.name}; prefer { _TIER0_NAMES[0] }.",
            stacklevel=2,
        )
    if tier1_path is not None and tier1_path.name != _TIER1_NAMES[0]:
        warnings.warn(
            f"Using legacy tier1 filename {tier1_path.name}; prefer { _TIER1_NAMES[0] }.",
            stacklevel=2,
        )
    if per_sample_path is not None and per_sample_path.name != _PER_SAMPLE_NAMES[0]:
        warnings.warn(
            f"Using legacy per-sample filename {per_sample_path.name}; prefer { _PER_SAMPLE_NAMES[0] }.",
            stacklevel=2,
        )

    return {
        "tier0": _load_json(tier0_path) if tier0_path is not None else None,
        "tier1": _load_json(tier1_path) if tier1_path is not None else None,
        "per_sample": _load_jsonl(per_sample_path) if per_sample_path is not None else None,
        "metrics_summary": _load_schema_json(metrics_summary_path, METRICS_SUMMARY_SCHEMA_VERSION)
        if metrics_summary_path is not None
        else None,
        "failure_breakdown": _load_schema_json(
            failure_breakdown_path, FAILURE_BREAKDOWN_SCHEMA_VERSION
        )
        if failure_breakdown_path is not None
        else None,
        "paths": {
            "tier0": str(tier0_path) if tier0_path is not None else None,
            "tier1": str(tier1_path) if tier1_path is not None else None,
            "per_sample": str(per_sample_path) if per_sample_path is not None else None,
            "metrics_summary": str(metrics_summary_path) if metrics_summary_path is not None else None,
            "failure_breakdown": str(failure_breakdown_path)
            if failure_breakdown_path is not None
            else None,
        },
    }
