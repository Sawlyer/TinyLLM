"""Memory estimates for autoregressive key/value caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from tinyllm.config.schema import ModelConfig


@dataclass(frozen=True, slots=True)
class KVCacheMemoryEstimate:
    batch_size: int
    context_length: int
    dtype: str
    bytes_per_element: int
    bytes_per_layer: int
    total_bytes: int

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1024**2)

    def to_dict(self) -> dict[str, int | float | str]:
        return {**asdict(self), "total_mib": self.total_mib}


def estimate_kv_cache_memory(
    model_config: ModelConfig,
    *,
    batch_size: int,
    context_length: int,
    dtype: torch.dtype,
) -> KVCacheMemoryEstimate:
    """Estimate preallocated key/value tensor bytes, excluding all other memory."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (
        type(context_length) is not int
        or context_length <= 0
        or context_length > model_config.max_seq_len
    ):
        raise ValueError(
            f"context_length must be between 1 and model max_seq_len ({model_config.max_seq_len})"
        )
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
    if dtype not in supported_dtypes:
        raise ValueError("dtype must be float16, bfloat16, float32, or float64")

    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    head_dim = model_config.d_model // model_config.n_heads
    bytes_per_layer = (
        2
        * batch_size
        * model_config.n_kv_heads
        * head_dim
        * context_length
        * bytes_per_element
    )
    return KVCacheMemoryEstimate(
        batch_size=batch_size,
        context_length=context_length,
        dtype=str(dtype).removeprefix("torch."),
        bytes_per_element=bytes_per_element,
        bytes_per_layer=bytes_per_layer,
        total_bytes=bytes_per_layer * model_config.n_layers,
    )


def require_available_kv_cache_memory(
    estimate: KVCacheMemoryEstimate,
    *,
    available_bytes: int,
) -> None:
    """Raise an actionable error when estimated cache bytes exceed available memory."""
    if type(available_bytes) is not int or available_bytes < 0:
        raise ValueError("available_bytes must be a non-negative integer")
    if estimate.total_bytes > available_bytes:
        raise MemoryError(
            f"KV cache needs {estimate.total_bytes} bytes (~{estimate.total_mib:.1f} MiB; "
            f"{estimate.dtype}, batch {estimate.batch_size}, context {estimate.context_length}), "
            f"but only {available_bytes} bytes (~{available_bytes / (1024**2):.1f} MiB) "
            "are available. Reduce model.max_seq_len, "
            "reduce batch size, use a lower-precision model, or pass --no-cache."
        )
