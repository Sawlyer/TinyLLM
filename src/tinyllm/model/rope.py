"""Rotary positional embeddings."""

import torch
from torch import Tensor, nn

_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


class RotaryEmbedding(nn.Module):
    """Apply Llama-style rotary embeddings to attention heads."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("dim must be a positive even number")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if theta <= 0:
            raise ValueError("theta must be positive")

        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inv_freq)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Rotate ``x`` shaped ``[batch, heads, sequence, head_dim]``."""
        if x.ndim != 4:
            raise ValueError("x must have four dimensions: [batch, heads, sequence, head_dim]")
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}")
        if positions.ndim not in (1, 2):
            raise ValueError("positions must have shape [sequence] or [batch, sequence]")
        if positions.dtype not in _INTEGER_DTYPES:
            raise TypeError("positions must have an integer dtype")
        if positions.shape[-1] != x.shape[-2]:
            raise ValueError("positions length must match the sequence dimension")
        if positions.ndim == 2 and positions.shape[0] != x.shape[0]:
            raise ValueError("batched positions must match the input batch size")

        indices = positions.to(device=x.device, dtype=torch.long)
        torch._assert_async(((indices >= 0) & (indices < self.max_seq_len)).all())

        cos = self.cos_cached[indices].to(dtype=x.dtype)
        sin = self.sin_cached[indices].to(dtype=x.dtype)
        if positions.ndim == 1:
            while cos.ndim < x.ndim:
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
        else:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        return x * cos + self._rotate_half(x) * sin
