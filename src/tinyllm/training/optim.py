from __future__ import annotations

import math

from torch import nn
from torch.optim import AdamW

from tinyllm.model.norm import RMSNorm

_NORMALIZATION_TYPES = (
    RMSNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
)


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    *,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> AdamW:
    """Build AdamW with bias and normalization parameters excluded from decay."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        normalization = isinstance(module, _NORMALIZATION_TYPES) or class_name.endswith("norm")
        for name, parameter in module.named_parameters(recurse=False):
            identity = id(parameter)
            if not parameter.requires_grad or identity in seen:
                continue
            seen.add(identity)
            if name == "bias" or normalization:
                no_decay.append(parameter)
            else:
                decay.append(parameter)

    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return AdamW(groups, lr=learning_rate, betas=betas, eps=eps)


def cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Return linear-warmup, cosine-decay learning rate for an optimizer step."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    if type(warmup_steps) is not int or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be an integer in [0, total_steps)")
    if not math.isfinite(max_lr) or max_lr <= 0:
        raise ValueError("max_lr must be finite and positive")
    if not math.isfinite(min_lr) or not 0 <= min_lr <= max_lr:
        raise ValueError("min_lr must be finite and in [0, max_lr]")

    if warmup_steps and step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine
