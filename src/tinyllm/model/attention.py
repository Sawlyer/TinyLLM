"""Causal grouped-query self-attention."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.config.schema import ModelConfig
from tinyllm.model.cache import KVCache
from tinyllm.model.rope import RotaryEmbedding


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with fewer key/value than query heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.max_seq_len = config.max_seq_len
        self.head_dim = config.d_model // config.n_heads
        self.queries_per_kv = config.n_heads // config.n_kv_heads
        if self.head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")

        kv_width = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_width, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_width, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_seq_len,
            config.rope_theta,
        )
        self.attention_dropout = config.dropout
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache | None]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [batch, sequence, {self.d_model}]")
        if positions.ndim not in (1, 2) or positions.shape[-1] != x.shape[1]:
            raise ValueError("positions must match the input sequence length")
        if positions.ndim == 2 and positions.shape[0] != x.shape[0]:
            raise ValueError("batched positions must match the input batch size")

        batch_size, sequence_length, _ = x.shape

        # q: [batch, query_heads, sequence, head_dim]
        query = self.q_proj(x).view(batch_size, sequence_length, self.n_heads, self.head_dim)
        query = query.transpose(1, 2)
        # k/v: [batch, kv_heads, sequence, head_dim]
        key = self.k_proj(x).view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
        value = self.v_proj(x).view(batch_size, sequence_length, self.n_kv_heads, self.head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.rope(query, positions)
        key = self.rope(key, positions)

        cached_length = 0 if cache is None else cache.length
        updated_cache = None
        if cache is not None:
            cache.reserve(self.max_seq_len)
            updated_cache = cache.append(key, value)
            if updated_cache.key is None or updated_cache.value is None:
                raise RuntimeError("cache append did not store key/value tensors")
            key = updated_cache.key
            value = updated_cache.value

        # Treat each KV head as a batch item and its query group as SDPA heads.
        # Expanded k/v use stride-zero views instead of materialized copies.
        grouped_batch = batch_size * self.n_kv_heads
        key_sequence_length = key.shape[2]
        query = query.reshape(
            batch_size,
            self.n_kv_heads,
            self.queries_per_kv,
            sequence_length,
            self.head_dim,
        ).reshape(grouped_batch, self.queries_per_kv, sequence_length, self.head_dim)
        key = key.reshape(grouped_batch, 1, key_sequence_length, self.head_dim).expand(
            -1, self.queries_per_kv, -1, -1
        )
        value = value.reshape(grouped_batch, 1, key_sequence_length, self.head_dim).expand(
            -1, self.queries_per_kv, -1, -1
        )

        attention_mask = None
        if cache is not None:
            query_positions = cached_length + torch.arange(sequence_length, device=x.device)
            key_positions = torch.arange(key_sequence_length, device=x.device)
            attention_mask = key_positions[None, :] <= query_positions[:, None]

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=cache is None,
        )
        # [batch, query_heads, sequence, head_dim] -> [batch, sequence, d_model]
        attended = attended.reshape(batch_size, self.n_heads, sequence_length, self.head_dim)
        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        )
        return self.residual_dropout(self.out_proj(attended)), updated_cache
