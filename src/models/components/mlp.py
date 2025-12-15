from typing import Optional

import torch
from torch import nn

class TinyMLP(nn.Module):
    """Simple MLP backbone over flattened feature vectors."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
