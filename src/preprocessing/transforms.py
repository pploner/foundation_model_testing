# src/preprocessing/transforms.py

import torch
import torch.nn.functional as F
from .utils import CATEGORY_MAP  # hardcoded PID, charge, btag categories

##########################################
# Transformation Functions
##########################################

def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input unchanged."""
    return x

def log1p_signed(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    log1p(abs(x)) * sign(x), mask-aware (ignores zeros / padding).
    """
    mask = x != 0.0
    x_new = torch.zeros_like(x)
    if mask.any():
        x_new[mask] = torch.log1p(torch.abs(x[mask]) + eps) * torch.sign(x[mask])
    return x_new

def trig_phi(x: torch.Tensor) -> torch.Tensor:
    """
    Expand phi into sin(phi), cos(phi), masking padded zeros.
    Input: [N] or [N, 1]
    Output: [N, 2]
    """
    # Create mask for valid phi entries
    mask = x != 0.0

    # Compute sin and cos
    sin_x = torch.zeros_like(x)
    cos_x = torch.zeros_like(x)

    if mask.any():
        sin_x[mask] = torch.sin(x[mask])
        cos_x[mask] = torch.cos(x[mask])

    # Stack into [N, 2]
    out = torch.stack((sin_x, cos_x), dim=-1)
    return out

def one_hot_encode(x: torch.Tensor, category_type: str) -> torch.Tensor:
    """
    One-hot encode based on hardcoded categories.
    """
    categories = CATEGORY_MAP[category_type]
    cat_to_index = {cat: idx for idx, cat in enumerate(categories)}

    # Map values to indices
    # NOTE: x may be shape [batch], so iterate element-wise
    idx_tensor = torch.tensor(
        [cat_to_index[int(v)] for v in x],
        dtype=torch.long,
        device=x.device
    )
    return F.one_hot(idx_tensor, num_classes=len(categories)).float()

##########################################
# Transform Registry
##########################################

TRANSFORM_REGISTRY = {
    "identity": identity,
    "log1p": log1p_signed,
    "trig": trig_phi,
    "onehot": one_hot_encode,
}

##########################################
# Dispatcher
##########################################

def apply_transform(x: torch.Tensor, transform_name: str, group_name: str = None) -> torch.Tensor:
    """
    Apply a transform using the registry.
    For onehot, group_name must be provided to select category mapping.
    """
    if transform_name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform '{transform_name}'. Available: {list(TRANSFORM_REGISTRY.keys())}")

    fn = TRANSFORM_REGISTRY[transform_name]

    if transform_name == "onehot":
        if group_name not in CATEGORY_MAP:
            raise ValueError(f"Onehot transform requested for group '{group_name}', but no CATEGORY_MAP defined.")
        return fn(x, group_name)
    else:
        return fn(x)
