from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _hash_key(row: Dict) -> str:
    parts = [
        str(row.get("formula", "")),
        str(row.get("composition", "")),
        str(row.get("cif_path", "")),
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SUN metrics.")
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stable-key", type=str, default="formation_pass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_jsonl(args.per_sample)
    if not rows:
        args.out.write_text(json.dumps({"status": "empty"}, indent=2), encoding="utf-8")
        return

    stable_flags: List[int] = []
    hashes: List[str] = []
    for row in rows:
        stable_flags.append(int(bool(row.get(args.stable_key))))
        hashes.append(_hash_key(row))

    unique_count = len(set(hashes))
    stable_count = int(np.sum(stable_flags))
    total = len(rows)
    result = {
        "status": "ok",
        "total_samples": total,
        "stable_count": stable_count,
        "unique_count": unique_count,
        "novel_count": unique_count,  # placeholder until reference set provided
        "stable_rate": stable_count / max(total, 1),
        "unique_rate": unique_count / max(total, 1),
        "novel_rate": unique_count / max(total, 1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved SUN metrics to {args.out}")


if __name__ == "__main__":
    main()
