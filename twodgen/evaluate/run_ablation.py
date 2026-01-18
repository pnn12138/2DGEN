from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load ablation matrix and emit a run plan.")
    parser.add_argument("--matrix", type=Path, required=True, help="Ablation matrix json.")
    parser.add_argument("--out", type=Path, required=True, help="Output plan json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    experiments: List[Dict[str, Any]] = matrix.get("experiments", [])
    for exp in experiments:
        exp.setdefault("status", "planned")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiments": experiments}, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved ablation plan to {args.out}")


if __name__ == "__main__":
    main()
