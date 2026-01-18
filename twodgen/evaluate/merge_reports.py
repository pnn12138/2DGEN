from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


FIELD_ALIAS_MAP = {
    "id": ["sample_id", "index"],
    "cif_path": ["path", "cif"],
    "formation_energy_per_atom": ["formation_energy"],
}


def _load_jsonl(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row = _normalize_row(row)
            key = str(row.get("id"))
            rows[key] = row
    return rows


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    for target, aliases in FIELD_ALIAS_MAP.items():
        if target not in normalized:
            for alias in aliases:
                if alias in normalized:
                    normalized[target] = normalized[alias]
                    break
    return normalized


def _merge_row(base: Dict[str, Any], extra: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key in ("id", "cif_path"):
            continue
        if key in merged and merged[key] != value:
            merged[f"{prefix}_{key}"] = value
        else:
            merged[key] = value
    return merged


def _compute_rates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _rate(key: str) -> Optional[float]:
        vals = [int(bool(r.get(key))) for r in rows if key in r]
        if not vals:
            return None
        return float(np.mean(vals))

    return {
        "valid_rate": _rate("valid"),
        "cond_match_rate": _rate("cond_match"),
        "formation_pass_rate": _rate("formation_pass"),
        "property_pass_rate": _rate("property_pass"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-sample outputs into a unified report.")
    parser.add_argument("--tier0", type=Path, required=True, help="per_sample_tier0.jsonl")
    parser.add_argument("--conditions", type=Path, default=None, help="per_sample_conditions.jsonl")
    parser.add_argument("--energy", type=Path, default=None, help="per_sample_energy.jsonl")
    parser.add_argument("--formation", type=Path, default=None, help="per_sample_formation.jsonl")
    parser.add_argument("--property", type=Path, default=None, help="per_sample_property.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tier0_rows = _load_jsonl(args.tier0)
    cond_rows = _load_jsonl(args.conditions)
    energy_rows = _load_jsonl(args.energy)
    formation_rows = _load_jsonl(args.formation)
    property_rows = _load_jsonl(args.property)

    merged_rows: List[Dict[str, Any]] = []
    for key, base in tier0_rows.items():
        row = dict(base)
        if key in cond_rows:
            row = _merge_row(row, cond_rows[key], "conditions")
        if key in energy_rows:
            row = _merge_row(row, energy_rows[key], "energy")
        if key in formation_rows:
            row = _merge_row(row, formation_rows[key], "formation")
        if key in property_rows:
            row = _merge_row(row, property_rows[key], "property")
        merged_rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.out_dir / "per_sample.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "schema_version": "merged_report_v1",
        "total_samples": len(merged_rows),
        "rates": _compute_rates(merged_rows),
        "inputs": {
            "tier0": str(args.tier0),
            "conditions": str(args.conditions) if args.conditions else None,
            "energy": str(args.energy) if args.energy else None,
            "formation": str(args.formation) if args.formation else None,
            "property": str(args.property) if args.property else None,
        },
    }
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved merged report to {args.out_dir}")


if __name__ == "__main__":
    main()
