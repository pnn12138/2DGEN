from __future__ import annotations

import json
import warnings
from pathlib import Path

from twodgen.evaluate.io import load_eval_outputs


def test_load_eval_outputs_legacy_names(tmp_path: Path) -> None:
    (tmp_path / "tier0_metric.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (tmp_path / "tier1_metrics.json").write_text(json.dumps({"rate": 0.5}), encoding="utf-8")
    (tmp_path / "per_sanmple.jsonl").write_text(
        json.dumps({"sample": 1}) + "\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = load_eval_outputs(tmp_path)

    assert out["tier0"]["ok"] is True
    assert out["tier1"]["rate"] == 0.5
    assert out["per_sample"][0]["sample"] == 1
    assert out["paths"]["tier0"].endswith("tier0_metric.json")
    assert out["paths"]["tier1"].endswith("tier1_metrics.json")
    assert out["paths"]["per_sample"].endswith("per_sanmple.jsonl")
    assert any("legacy" in str(warning.message) for warning in caught)
