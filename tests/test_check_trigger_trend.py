from __future__ import annotations

import json
from pathlib import Path

from twodgen.evaluate.check_trigger_trend import main


def test_check_trigger_trend_pass(tmp_path: Path, monkeypatch) -> None:
    metrics = tmp_path / "train_metrics.jsonl"
    rows = [
        {
            "step": 0,
            "post_project_trigger_rate_train_proxy": 0.9,
            "cond_violation_rate_train_proxy": 0.8,
            "vacuum_violation_rate_train_proxy": 0.7,
        },
        {
            "step": 1,
            "post_project_trigger_rate_train_proxy": 0.6,
            "cond_violation_rate_train_proxy": 0.5,
            "vacuum_violation_rate_train_proxy": 0.4,
        },
        {
            "step": 2,
            "post_project_trigger_rate_train_proxy": 0.3,
            "cond_violation_rate_train_proxy": 0.2,
            "vacuum_violation_rate_train_proxy": 0.1,
        },
    ]
    with metrics.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        ["check_trigger_trend", "--metrics-jsonl", str(metrics), "--out", str(out)],
    )
    main()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["trend"]["post_project_trigger_rate_train_proxy"]["improved"] is True

