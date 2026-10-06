"""Root mean square normalization."""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Normalize activations by their root mean square."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.weight.numel():
            raise ValueError(f"expected final dimension {self.weight.numel()}, got {x.shape[-1]}")
        reduction_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        mean_square = x.to(reduction_dtype).square().mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(mean_square + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)
