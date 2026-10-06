"""Token sampling controls for autoregressive inference."""

import torch
from torch import Tensor


def sample_next(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float,
    generator: torch.Generator,
) -> Tensor:
    """Select one token per batch row using greedy or filtered sampling."""
    if logits.ndim != 2 or logits.shape[-1] == 0:
        raise ValueError("logits must have shape [batch, vocabulary]")
    if bool(torch.isnan(logits).any()):
        raise ValueError("logits contain NaN values; check model weights and inference precision")
    if bool(torch.isposinf(logits).any()):
        raise ValueError("logits contain positive infinity; check inference precision")
    if not bool(torch.isfinite(logits).any(dim=-1).all()):
        raise ValueError("each batch row must contain at least one finite logit")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    filtered = logits / temperature
    vocabulary_size = filtered.shape[-1]
    if top_k is not None and top_k < vocabulary_size:
        top_values, top_indices = torch.topk(filtered, top_k, dim=-1)
        filtered = torch.full_like(filtered, float("-inf")).scatter(
            -1,
            top_indices,
            top_values,
        )

    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative >= top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
        filtered = filtered.masked_fill(remove, float("-inf"))

    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)
