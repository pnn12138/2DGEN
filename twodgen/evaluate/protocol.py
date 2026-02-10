from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class EvalProtocol:
    name: str
    num_samples: int
    seeds: List[int]


PROTOCOLS: Dict[str, EvalProtocol] = {
    "quick": EvalProtocol(name="quick", num_samples=2000, seeds=[0, 1, 2]),
    "final": EvalProtocol(name="final", num_samples=20000, seeds=[0, 1, 2]),
}


def get_protocol(name: str) -> EvalProtocol:
    key = str(name).strip().lower()
    if key not in PROTOCOLS:
        supported = ", ".join(sorted(PROTOCOLS))
        raise ValueError(f"Unknown protocol '{name}'. Supported: {supported}.")
    return PROTOCOLS[key]

