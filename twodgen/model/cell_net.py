from __future__ import annotations

import torch
from torch import nn


class CellNet(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int = 128, out_dim: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)
