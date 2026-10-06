"""Transformer feed-forward network."""

import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.config.schema import ModelConfig


class SwiGLU(nn.Module):
    """Silu-gated feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = int(config.d_model * config.mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("mlp hidden dimension must be positive")
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))
