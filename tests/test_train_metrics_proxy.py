from __future__ import annotations

import json
from pathlib import Path

import torch

from twodgen.scrip.train_tokens import _aggregate_train_metrics_jsonl, _compute_train_proxy_metrics


def test_compute_train_proxy_metrics() -> None:
    metrics = {
        "cond_lattice_violation_rate": torch.tensor(0.2),
        "vacuum_gap": torch.tensor([0.0, 0.5, 1.0]),
        "pred_angle_out_rate": torch.tensor(0.1),
    }
    out = _compute_train_proxy_metrics(metrics, collision_rate=0.3)
    assert abs(out["cond_violation_rate_train_proxy"] - 0.2) < 1e-8
    assert abs(out["vacuum_violation_rate_train_proxy"] - (2.0 / 3.0)) < 1e-6
    assert abs(out["post_project_trigger_rate_train_proxy"] - (2.0 / 3.0)) < 1e-6


def test_aggregate_train_metrics_jsonl_proxy_trend(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    rows = [
        {"step": 0, "post_project_trigger_rate_train_proxy": 0.8},
        {"step": 1, "post_project_trigger_rate_train_proxy": 0.6},
        {"step": 2, "post_project_trigger_rate_train_proxy": 0.4},
        {"step": 3, "post_project_trigger_rate_train_proxy": 0.2},
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    out = _aggregate_train_metrics_jsonl(path)
    trend = out["proxy_trend"]["post_project_trigger_rate_train_proxy"]
    assert trend["available"] is True
    assert trend["improved"] is True
    assert trend["second_half_mean"] < trend["first_half_mean"]
