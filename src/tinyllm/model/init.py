"""Weight initialization for TinyLLM."""

import math

import torch
from torch import nn

BASE_INIT_STD = 0.02


def initialize_weights(module: nn.Module) -> None:
    """Initialize ordinary learned projections and embeddings."""
    if isinstance(module, (nn.Embedding, nn.Linear)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=BASE_INIT_STD)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def initialize_residual_projection(module: nn.Linear, n_layers: int) -> None:
    """Scale residual projection variance by total network depth."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    std = BASE_INIT_STD / math.sqrt(2 * n_layers)
    torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
