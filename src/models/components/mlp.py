import torch
from torch import nn
from typing import Optional

class Standardize(nn.Module):
    """Apply (x - mean) / std safely; falls back to identity if no stats given."""
    def __init__(self, mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None, eps: float = 1e-6):
        super().__init__()
        if mean is not None:
            self.register_buffer("mean", mean.float())
        else:
            self.mean = None
        if std is not None:
            self.register_buffer("std", std.float())
        else:
            self.std = None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / (self.std + self.eps)

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