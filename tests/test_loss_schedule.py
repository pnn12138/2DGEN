from __future__ import annotations

import numpy as np

from twodgen.loss.schedule import LossWeightScheduleConfig, LossWeightScheduler


def test_loss_weight_scheduler_linear_scales_selected_keys() -> None:
    base = {"vacuum": 1.0, "cond": 2.0, "other": 3.0}
    cfg = LossWeightScheduleConfig(
        warmup_steps=10,
        start_factor=0.0,
        end_factor=1.0,
        keys=("vacuum", "cond"),
        schedule="linear",
    )
    scheduler = LossWeightScheduler(base, cfg)

    w0 = scheduler.weights(0)
    w9 = scheduler.weights(9)

    assert np.isclose(w0["vacuum"], base["vacuum"] * 0.1)
    assert np.isclose(w0["cond"], base["cond"] * 0.1)
    assert np.isclose(w0["other"], base["other"])

    assert np.isclose(w9["vacuum"], base["vacuum"])
    assert np.isclose(w9["cond"], base["cond"])
    assert np.isclose(w9["other"], base["other"])
