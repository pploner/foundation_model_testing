# src/preprocessing/normalization.py

import torch
from typing import Dict, Optional

##########################################
# Base Class
##########################################

class NormalizerBase:
    def fit(self, x: torch.Tensor):
        raise NotImplementedError

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        self.fit(x)
        return self.transform(x)

    def state_dict(self) -> Dict:
        raise NotImplementedError

    def load_state_dict(self, state: Dict):
        raise NotImplementedError


##########################################
# Standard Normalization: (x - mean) / std
##########################################

class StandardScaler(NormalizerBase):
    def __init__(self, epsilon: float = 1e-6):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.epsilon = epsilon

    def fit(self, x: torch.Tensor):
        mask = x != 0.0
        valid = x[mask]
        if valid.numel() > 0:
            self.mean = valid.mean(dim=0)
            self.std = valid.std(dim=0) + self.epsilon
        else:
            self.mean = torch.tensor(0.0, device=x.device)
            self.std = torch.tensor(1.0, device=x.device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mask = x != 0.0
        x_new = torch.zeros_like(x)
        if mask.any():
            x_new[mask] = (x[mask] - self.mean) / self.std
        return x_new

    def state_dict(self) -> Dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state: Dict):
        self.mean = state["mean"]
        self.std = state["std"]


##########################################
# Robust Normalization: (x - median) / IQR
##########################################

class RobustScaler(NormalizerBase):
    def __init__(self, epsilon: float = 1e-6):
        self.median: Optional[torch.Tensor] = None
        self.iqr: Optional[torch.Tensor] = None
        self.epsilon = epsilon

    def fit(self, x: torch.Tensor):
        mask = x != 0.0
        valid = x[mask]
        if valid.numel() > 0:
            self.median = valid.median(dim=0).values
            q75 = torch.quantile(valid, 0.75, dim=0)
            q25 = torch.quantile(valid, 0.25, dim=0)
            self.iqr = (q75 - q25).clamp_min(self.epsilon)
        else:
            self.median = torch.tensor(0.0, device=x.device)
            self.iqr = torch.tensor(1.0, device=x.device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        mask = x != 0.0
        x_new = torch.zeros_like(x)
        if mask.any():
            x_new[mask] = (x[mask] - self.median) / self.iqr
        return x_new

    def state_dict(self) -> Dict:
        return {"median": self.median, "iqr": self.iqr}

    def load_state_dict(self, state: Dict):
        self.median = state["median"]
        self.iqr = state["iqr"]


##########################################
# Min-Max Normalization: (x - min) / (max - min)
##########################################

class MinMaxScaler(NormalizerBase):
    def __init__(self, epsilon: float = 1e-6):
        self.min = None
        self.max = None
        self.epsilon = epsilon

    def fit(self, x: torch.Tensor):
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min) / (self.max - self.min + self.epsilon)

    def state_dict(self) -> Dict:
        return {"min": self.min, "max": self.max}

    def load_state_dict(self, state: Dict):
        self.min = state["min"]
        self.max = state["max"]

##########################################
# Identity Normalization (None)
##########################################

class IdentityScaler(NormalizerBase):
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def fit(self, x: torch.Tensor):
        pass

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def state_dict(self) -> Dict:
        return {}

    def load_state_dict(self, state: Dict):
        pass


##########################################
# Normalizer Registry
##########################################

NORMALIZER_REGISTRY = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "minmax": MinMaxScaler,
    "none": IdentityScaler,
}

def create_normalizer(name: str, epsilon: float = 1e-6) -> NormalizerBase:
    if name not in NORMALIZER_REGISTRY:
        raise ValueError(f"Unknown normalizer '{name}'. Available: {list(NORMALIZER_REGISTRY.keys())}")
    return NORMALIZER_REGISTRY[name](epsilon=epsilon)
