from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class InferenceBenchmarkResult:
    mode: str
    mean_loss: float
    perplexity: float
    tokens_per_second: tuple[float, ...]
    peak_allocated_bytes: int
    mean_absolute_logit_error: float
    max_absolute_logit_error: float


def benchmark_inference(
    model: nn.Module,
    batches: Sequence[tuple[Tensor, Tensor]],
    *,
    mode: str,
    device: torch.device,
    autocast_context: Callable[[], AbstractContextManager[object]],
    reference_logits: Sequence[Tensor],
    repetitions: int = 3,
    warmup_repetitions: int = 1,
    timer: Callable[[], float] = time.perf_counter,
) -> InferenceBenchmarkResult:
    if not batches:
        raise ValueError("batches must not be empty")
    if len(reference_logits) != len(batches):
        raise ValueError("reference_logits must match batches")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if warmup_repetitions < 0:
        raise ValueError("warmup_repetitions must be non-negative")

    model.eval()
    losses: list[float] = []
    absolute_error_sum = 0.0
    absolute_error_count = 0
    maximum_error = 0.0
    throughputs: list[float] = []
    total_tokens = sum(inputs.numel() for inputs, _ in batches)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup_repetitions):
            for cpu_inputs, cpu_targets in batches:
                with autocast_context():
                    model(cpu_inputs.to(device), cpu_targets.to(device))
        for repetition in range(repetitions):
            _synchronize(device)
            started = timer()
            for batch_index, (cpu_inputs, cpu_targets) in enumerate(batches):
                inputs = cpu_inputs.to(device)
                targets = cpu_targets.to(device)
                with autocast_context():
                    output = model(inputs, targets)
                if output.loss is None or not bool(torch.isfinite(output.loss)):
                    raise FloatingPointError(f"non-finite {mode} inference loss")
                if repetition == 0:
                    losses.append(float(output.loss))
                    difference = output.logits.float().cpu() - reference_logits[batch_index].float()
                    absolute = difference.abs()
                    absolute_error_sum += float(absolute.sum())
                    absolute_error_count += absolute.numel()
                    maximum_error = max(maximum_error, float(absolute.max()))
            _synchronize(device)
            elapsed = timer() - started
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("timer must report positive finite elapsed time")
            throughputs.append(total_tokens / elapsed)

    mean_loss = sum(losses) / len(losses)
    peak_allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return InferenceBenchmarkResult(
        mode=mode,
        mean_loss=mean_loss,
        perplexity=math.exp(min(mean_loss, 80.0)),
        tokens_per_second=tuple(throughputs),
        peak_allocated_bytes=peak_allocated,
        mean_absolute_logit_error=absolute_error_sum / absolute_error_count,
        max_absolute_logit_error=maximum_error,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
