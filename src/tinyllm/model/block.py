"""Pre-normalized Transformer decoder block."""

from torch import Tensor, nn

from tinyllm.config.schema import ModelConfig
from tinyllm.model.attention import GroupedQueryAttention
from tinyllm.model.cache import KVCache
from tinyllm.model.mlp import SwiGLU
from tinyllm.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Apply causal attention and feed-forward residual updates."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.feed_forward_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache | None]:
        attended, updated_cache = self.attention(
            self.attention_norm(x),
            positions,
            cache,
        )
        x = x + attended
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x, updated_cache
