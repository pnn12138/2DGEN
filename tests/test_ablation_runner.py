from __future__ import annotations

import math

from twodgen.evaluate.ablation_runner import _aggregate_by_variant, _variant_args


def test_variant_args_contains_projection_flags() -> None:
    baseline = _variant_args("baseline", cond_max=40.0)
    full = _variant_args("full_projection", cond_max=40.0)
    assert "--no-post-project" in baseline
    assert "--post-project" in full
    assert "angle,cond,inplane,volume" in full


def test_aggregate_by_variant_delta() -> None:
    rows = [
        {"variant": "baseline", "success_geom_rate": 0.20, "bad_volume_rate": 0.5},
        {"variant": "baseline", "success_geom_rate": 0.30, "bad_volume_rate": 0.4},
        {"variant": "full_projection", "success_geom_rate": 0.50, "bad_volume_rate": 0.2},
        {"variant": "full_projection", "success_geom_rate": 0.60, "bad_volume_rate": 0.1},
    ]
    out = _aggregate_by_variant(rows)
    assert out["variants"]["baseline"]["success_geom_rate"]["mean"] == 0.25
    assert out["variants"]["full_projection"]["success_geom_rate"]["mean"] == 0.55
    assert math.isclose(out["delta_success_geom_rate_full_minus_baseline"], 0.30, rel_tol=1e-9)
