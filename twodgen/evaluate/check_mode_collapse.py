from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _coverage_scalar(data: Dict[str, Any]) -> float:
    vals = []
    space_cov = data.get("spacegroup", {}).get("coverage_vs_230")
    if isinstance(space_cov, (int, float)):
        vals.append(float(space_cov))
    n_cov = data.get("n_atoms_coverage", {}).get("coverage")
    if isinstance(n_cov, (int, float)):
        vals.append(float(n_cov))
    l_cov = data.get("lattice_coverage", {}).get("coverage")
    if isinstance(l_cov, (int, float)):
        vals.append(float(l_cov))
    comp_cov = data.get("composition_coverage", {}).get("relative_coverage")
    if isinstance(comp_cov, (int, float)):
        vals.append(float(comp_cov))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _novelty_median(path: Optional[Path]) -> Optional[float]:
    if path is None:
        return None
    data = _read_json(path)
    novelty = data.get("novelty", {})
    val = novelty.get("median")
    if isinstance(val, (int, float)):
        return float(val)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check diversity/novelty collapse against baseline.")
    parser.add_argument("--baseline-diversity", type=Path, required=True)
    parser.add_argument("--current-diversity", type=Path, required=True)
    parser.add_argument("--baseline-novelty", type=Path, default=None)
    parser.add_argument("--current-novelty", type=Path, default=None)
    parser.add_argument("--max-coverage-drop", type=float, default=0.20)
    parser.add_argument("--max-novelty-drop", type=float, default=0.10)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_div = _read_json(args.baseline_diversity)
    cur_div = _read_json(args.current_diversity)
    base_cov = _coverage_scalar(base_div)
    cur_cov = _coverage_scalar(cur_div)
    cov_drop = (base_cov - cur_cov) / max(base_cov, 1e-8)
    coverage_pass = bool(cov_drop <= float(args.max_coverage_drop))

    base_nov = _novelty_median(args.baseline_novelty)
    cur_nov = _novelty_median(args.current_novelty)
    novelty_drop = None
    novelty_pass = None
    if base_nov is not None and cur_nov is not None:
        novelty_drop = (base_nov - cur_nov) / max(base_nov, 1e-8)
        novelty_pass = bool(novelty_drop <= float(args.max_novelty_drop))

    checks = [coverage_pass]
    if novelty_pass is not None:
        checks.append(bool(novelty_pass))
    passed = all(checks)

    report = {
        "baseline_diversity": str(args.baseline_diversity),
        "current_diversity": str(args.current_diversity),
        "baseline_coverage": base_cov,
        "current_coverage": cur_cov,
        "coverage_drop_ratio": cov_drop,
        "coverage_pass": coverage_pass,
        "max_coverage_drop": float(args.max_coverage_drop),
        "baseline_novelty_median": base_nov,
        "current_novelty_median": cur_nov,
        "novelty_drop_ratio": novelty_drop,
        "novelty_pass": novelty_pass,
        "max_novelty_drop": float(args.max_novelty_drop),
        "pass": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved mode-collapse report to {args.out}")
    print(f"pass={passed}")


if __name__ == "__main__":
    main()

