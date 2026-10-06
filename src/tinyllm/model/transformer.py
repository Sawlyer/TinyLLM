"""Decoder-only TinyLLM language model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tinyllm.config.schema import ModelConfig
from tinyllm.model.block import TransformerBlock
from tinyllm.model.cache import KVCache
from tinyllm.model.init import initialize_residual_projection, initialize_weights
from tinyllm.model.norm import RMSNorm


@dataclass
class ModelOutput:
    """Language-model predictions and optional training/cache state."""

    logits: Tensor
    loss: Tensor | None = None
    caches: list[KVCache] | None = None


class TinyLLM(nn.Module):
    """Causal decoder-only Transformer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(initialize_weights)
        for block in self.blocks:
            initialize_residual_projection(block.attention.out_proj, config.n_layers)
            initialize_residual_projection(block.feed_forward.down_proj, config.n_layers)
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[0] == 0:
            raise ValueError("input batch must contain at least one sequence")
        sequence_length = input_ids.shape[1]
        if sequence_length == 0 or sequence_length > self.config.max_seq_len:
            raise ValueError(f"input sequence length must be in [1, {self.config.max_seq_len}]")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must match input_ids shape")
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError("caches must contain one cache per layer")

        position_offset = 0
        if caches is not None:
            if any(not isinstance(cache, KVCache) for cache in caches):
                raise TypeError("caches must contain KVCache instances")
            cache_lengths = {cache.length for cache in caches}
            if len(cache_lengths) != 1:
                raise ValueError("all layer caches must have the same sequence length")
            position_offset = next(iter(cache_lengths))
            if position_offset + sequence_length > self.config.max_seq_len:
                raise ValueError("cached sequence exceeds the model context length")

        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=input_ids.device,
        )
        hidden = self.token_embedding(input_ids)
        layer_caches: list[KVCache | None]
        layer_caches = [None] * len(self.blocks) if caches is None else list(caches)
        updated_caches: list[KVCache] | None = [] if caches is not None else None
        for block, cache in zip(self.blocks, layer_caches, strict=True):
            hidden, updated_cache = block(hidden, positions, cache)
            if updated_caches is not None:
                if updated_cache is None:
                    raise RuntimeError("attention did not return an updated cache")
                updated_caches.append(updated_cache)

        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
        return ModelOutput(
            logits=logits,
            loss=loss,
            caches=updated_caches,
        )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return unique total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
