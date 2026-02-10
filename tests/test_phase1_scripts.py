from __future__ import annotations

import json
from pathlib import Path

from twodgen.evaluate.collect_gscale_sweep import main as collect_gscale_main
from twodgen.scrip.sample_tokens import parse_args


def test_sample_tokens_parse_gscale_override() -> None:
    args = parse_args(
        [
            "--checkpoint",
            "dummy.pt",
            "--g-scale",
            "0.5",
            "--override-g-scale",
            "--num-samples",
            "1",
            "--steps",
            "1",
        ]
    )
    assert args.g_scale == 0.5
    assert args.override_g_scale is True


def test_collect_gscale_sweep(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    exp = runs_root / "E1_3_gscale_0p5" / "_aggregate"
    exp.mkdir(parents=True)
    (exp / "summary.json").write_text(
        json.dumps(
            {
                "variants": {
                    "full_projection": {
                        "success_geom_rate": {"mean": 0.4, "std": 0.01},
                        "valid_rate_eval": {"mean": 0.41},
                        "post_project_trigger_any_rate": {"mean": 0.2},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out = runs_root / "E1_3" / "_aggregate" / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "collect_gscale_sweep",
            "--runs-root",
            str(runs_root),
            "--experiment-prefix",
            "E1_3_gscale",
            "--g-scales",
            "0.5,1.0",
            "--out",
            str(out),
        ],
    )
    collect_gscale_main()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 2
    assert data["rows"][0]["available"] is True
    assert data["rows"][1]["available"] is False

