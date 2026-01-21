from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LossWeightScheduleConfig:
    warmup_steps: int = 15000
    start_factor: float = 0.0
    end_factor: float = 1.0
    keys: tuple[str, ...] = ("vacuum", "cond", "chol_bound", "expand_collision")
    schedule: str = "sigmoid"


class LossWeightScheduler:
    def __init__(self, base_weights: dict[str, float], cfg: LossWeightScheduleConfig) -> None:
        self._base_weights = dict(base_weights)
        self._cfg = cfg
        self._keys = set(cfg.keys)

    @staticmethod
    def _normalize_keys(keys: Iterable[str]) -> tuple[str, ...]:
        normalized = [key.strip().lower() for key in keys if key.strip()]
        return tuple(dict.fromkeys(normalized))

    def factor(self, step: int) -> float:
        if self._cfg.warmup_steps <= 0:
            return float(self._cfg.end_factor)
        progress = float(step + 1) / float(max(1, self._cfg.warmup_steps))
        progress = max(0.0, min(progress, 1.0))
        start = float(self._cfg.start_factor)
        end = float(self._cfg.end_factor)
        delta = end - start
        schedule = str(self._cfg.schedule).lower()
        if schedule == "sigmoid":
            import math

            mid = 0.5
            steepness = 12.0
            sigmoid = 1.0 / (1.0 + math.exp(-(progress - mid) * steepness))
            return start + delta * sigmoid
        if schedule == "cosine":
            import math

            cosine = 0.5 - 0.5 * math.cos(math.pi * progress)
            return start + delta * cosine
        return start + delta * progress

    def weights(self, step: int) -> dict[str, float]:
        factor = self.factor(step)
        out = dict(self._base_weights)
        for key in self._keys:
            if key in out:
                out[key] = out[key] * factor
        return out


__all__ = ["LossWeightScheduleConfig", "LossWeightScheduler"]
