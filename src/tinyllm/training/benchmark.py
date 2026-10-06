"""Deterministic, synchronized training precision benchmarks."""

from __future__ import annotations

import copy
import json
import math
import os
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from tinyllm.config.schema import ModelConfig
from tinyllm.errors import TinyLLMUserError
from tinyllm.model.transformer import TinyLLM
from tinyllm.training.capabilities import (
    Float8Backend,
    convert_linear_layers_to_fp8,
)
from tinyllm.training.precision import PrecisionPolicy, PrecisionUnavailableError

BENCHMARK_FORMAT_VERSION = 2
_PRECISION_MODES = frozenset({"bf16", "fp16", "fp32", "fp8"})


class BenchmarkExecutionError(TinyLLMUserError):
    """Declared benchmark-environment failure safe to record per mode."""


class PrecisionPolicyLike(Protocol):
    name: str

    def autocast(self): ...

    def backward(self, loss: Tensor) -> None: ...

    def step(self, optimizer: Optimizer) -> None: ...


class DeviceMetrics(Protocol):
    def clear_cache(self, device: torch.device) -> None: ...

    def reset_peak_memory_stats(self, device: torch.device) -> None: ...

    def synchronize(self, device: torch.device) -> None: ...

    def peak_allocated_bytes(self, device: torch.device) -> int: ...

    def peak_reserved_bytes(self, device: torch.device) -> int: ...


class TorchDeviceMetrics:
    """CUDA timing and allocator metrics, with explicit zero CPU VRAM."""

    def clear_cache(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def reset_peak_memory_stats(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def synchronize(self, device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def peak_allocated_bytes(self, device: torch.device) -> int:
        if device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_allocated(device))

    def peak_reserved_bytes(self, device: torch.device) -> int:
        if device.type != "cuda":
            return 0
        return int(torch.cuda.max_memory_reserved(device))


@dataclass(frozen=True, slots=True)
class PrecisionBenchmarkConfig:
    model: ModelConfig
    device: torch.device | str
    batch_size: int
    sequence_length: int
    warmup_steps: int
    measured_steps: int
    seed: int
    learning_rate: float
    weight_decay: float
    compile: bool = True
    output_path: Path | None = None
    overwrite_output: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")
        for name, value in (
            ("batch_size", self.batch_size),
            ("sequence_length", self.sequence_length),
            ("measured_steps", self.measured_steps),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        if self.sequence_length > self.model.max_seq_len:
            raise ValueError("sequence_length exceeds model.max_seq_len")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    mode: str
    seed: int
    batch_shape: tuple[int, int]
    warmup_steps: int
    measured_steps: int
    elapsed_seconds: float | None
    tokens_per_second: float | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    mean_loss: float | None
    finite_loss: bool | None
    status: Literal["ready", "unsupported", "failed"] = "ready"
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "status": self.status,
            "reason": self.reason,
            "seed": self.seed,
            "batch_shape": list(self.batch_shape),
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measured_steps,
            "elapsed_seconds": self.elapsed_seconds,
            "tokens_per_second": self.tokens_per_second,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "mean_loss": self.mean_loss,
            "finite_loss": self.finite_loss,
        }


def _default_model_factory(config: ModelConfig) -> nn.Module:
    return TinyLLM(config)


def _default_optimizer_factory(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _default_compiler(model: nn.Module) -> nn.Module:
    if not hasattr(torch, "compile"):
        raise RuntimeError("precision benchmark compilation requires torch.compile")
    return torch.compile(model)


def _default_batch_to_device(
    inputs: Tensor,
    targets: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    return inputs.to(device), targets.to(device)


def _default_loss_reader(losses: Tensor) -> list[float]:
    return losses.cpu().tolist()


@dataclass(slots=True)
class BenchmarkRuntime:
    model_factory: Callable[[ModelConfig], nn.Module] = _default_model_factory
    optimizer_factory: Callable[[nn.Module, float, float], Optimizer] = _default_optimizer_factory
    policy_factory: Callable[[str, torch.device], PrecisionPolicyLike] = PrecisionPolicy.create
    compiler: Callable[[nn.Module], nn.Module] = _default_compiler
    float8_backend: Float8Backend | None = None
    timer: Callable[[], float] = time.perf_counter
    device_metrics: DeviceMetrics = field(default_factory=TorchDeviceMetrics)
    batch_to_device: Callable[[Tensor, Tensor, torch.device], tuple[Tensor, Tensor]] = (
        _default_batch_to_device
    )
    loss_reader: Callable[[Tensor], list[float]] = _default_loss_reader


def benchmark_precision(
    config: PrecisionBenchmarkConfig,
    modes: list[str],
    *,
    runtime: BenchmarkRuntime | None = None,
) -> list[BenchmarkResult]:
    """Benchmark modes from identical model state and pre-generated batches."""
    if not isinstance(config, PrecisionBenchmarkConfig):
        raise TypeError("config must be a PrecisionBenchmarkConfig")
    normalized_modes = _validate_modes(modes)
    active_runtime = BenchmarkRuntime() if runtime is None else runtime
    device = torch.device(config.device)
    batches = _build_batches(config)

    torch.manual_seed(config.seed)
    base_model = active_runtime.model_factory(config.model)
    if not isinstance(base_model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    base_state = copy.deepcopy(base_model.state_dict())
    results: list[BenchmarkResult] = []
    for mode in normalized_modes:
        try:
            result = _benchmark_mode(
                config,
                mode,
                device,
                base_state,
                batches,
                active_runtime,
            )
        except PrecisionUnavailableError as error:
            result = _unmeasured_result(config, mode, "unsupported", str(error))
        except BenchmarkExecutionError as error:
            reason = f"{type(error).__name__}: {error}"
            result = _unmeasured_result(config, mode, "failed", reason)
        results.append(result)
    if config.output_path is not None:
        write_benchmark_results(
            config.output_path,
            results,
            overwrite=config.overwrite_output,
        )
    return results


def _unmeasured_result(
    config: PrecisionBenchmarkConfig,
    mode: str,
    status: Literal["unsupported", "failed"],
    reason: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        mode=mode,
        seed=config.seed,
        batch_shape=(config.batch_size, config.sequence_length),
        warmup_steps=config.warmup_steps,
        measured_steps=config.measured_steps,
        elapsed_seconds=None,
        tokens_per_second=None,
        peak_allocated_bytes=None,
        peak_reserved_bytes=None,
        mean_loss=None,
        finite_loss=None,
        status=status,
        reason=reason,
    )


def write_benchmark_results(
    path: Path,
    results: Sequence[BenchmarkResult],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist strict JSON benchmark output."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"benchmark output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "format_version": BENCHMARK_FORMAT_VERSION,
        "results": [result.to_dict() for result in results],
    }
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_modes(modes: list[str]) -> list[str]:
    if not isinstance(modes, list) or not modes:
        raise ValueError("modes must be a non-empty list")
    if not all(isinstance(mode, str) for mode in modes):
        raise ValueError("modes must contain precision names")
    normalized = [mode.lower() for mode in modes]
    if len(set(normalized)) != len(normalized):
        raise ValueError("modes must not contain duplicates")
    unsupported = set(normalized) - _PRECISION_MODES
    if unsupported:
        raise ValueError(f"modes contain unsupported precision: {sorted(unsupported)}")
    return normalized


def _build_batches(
    config: PrecisionBenchmarkConfig,
) -> list[tuple[Tensor, Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    return [
        _causal_batch(config, generator) for _ in range(config.warmup_steps + config.measured_steps)
    ]


def _causal_batch(
    config: PrecisionBenchmarkConfig,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    tokens = torch.randint(
        0,
        config.model.vocab_size,
        (config.batch_size, config.sequence_length + 1),
        generator=generator,
    )
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def _benchmark_mode(
    config: PrecisionBenchmarkConfig,
    mode: str,
    device: torch.device,
    base_state: dict[str, object],
    batches: list[tuple[Tensor, Tensor]],
    runtime: BenchmarkRuntime,
) -> BenchmarkResult:
    torch.manual_seed(config.seed)
    model = runtime.model_factory(config.model)
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    model.load_state_dict(base_state)
    model = model.to(device)
    policy = runtime.policy_factory(mode, device)
    if mode == "fp8":
        model = convert_linear_layers_to_fp8(
            model,
            backend=runtime.float8_backend,
            capability=getattr(policy, "fp8_capability", None),
        )
    optimizer = runtime.optimizer_factory(model, config.learning_rate, config.weight_decay)
    training_model = runtime.compiler(model) if config.compile or mode == "fp8" else model
    training_model.train()
    for cpu_inputs, cpu_targets in batches[: config.warmup_steps]:
        inputs, targets = runtime.batch_to_device(cpu_inputs, cpu_targets, device)
        _training_step(training_model, optimizer, policy, inputs, targets)
        del inputs, targets

    runtime.device_metrics.clear_cache(device)
    loss_buffer = torch.empty(config.measured_steps, device=device, dtype=torch.float32)
    runtime.device_metrics.reset_peak_memory_stats(device)
    runtime.device_metrics.synchronize(device)
    start = runtime.timer()
    for index, (cpu_inputs, cpu_targets) in enumerate(batches[config.warmup_steps :]):
        inputs, targets = runtime.batch_to_device(cpu_inputs, cpu_targets, device)
        loss = _training_step(training_model, optimizer, policy, inputs, targets)
        loss_buffer[index].copy_(loss)
        del inputs, targets, loss
    runtime.device_metrics.synchronize(device)
    elapsed = runtime.timer() - start
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise BenchmarkExecutionError("benchmark timer must report a positive finite duration")
    peak_allocated_bytes = runtime.device_metrics.peak_allocated_bytes(device)
    peak_reserved_bytes = runtime.device_metrics.peak_reserved_bytes(device)
    losses = runtime.loss_reader(loss_buffer)
    finite_loss = all(math.isfinite(loss) for loss in losses)
    mean_loss = sum(losses) / len(losses) if finite_loss else None
    measured_tokens = config.measured_steps * config.batch_size * config.sequence_length
    result = BenchmarkResult(
        mode=mode,
        seed=config.seed,
        batch_shape=(config.batch_size, config.sequence_length),
        warmup_steps=config.warmup_steps,
        measured_steps=config.measured_steps,
        elapsed_seconds=elapsed,
        tokens_per_second=measured_tokens / elapsed,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        mean_loss=mean_loss,
        finite_loss=finite_loss,
    )
    if not finite_loss:
        return replace(
            result,
            status="failed",
            reason="non-finite measured loss",
        )
    return result


def _training_step(
    model: nn.Module,
    optimizer: Optimizer,
    policy: PrecisionPolicyLike,
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    optimizer.zero_grad(set_to_none=True)
    with policy.autocast():
        output = model(inputs, targets)
    loss = getattr(output, "loss", None)
    if not isinstance(loss, Tensor) or loss.numel() != 1:
        raise BenchmarkExecutionError("benchmark model must return one scalar loss tensor")
    policy.backward(loss)
    _require_finite_values(
        [parameter.grad for parameter in model.parameters() if parameter.grad is not None],
        "gradient",
    )
    policy.step(optimizer)
    _require_finite_values(list(model.parameters()), "parameter")
    return loss.detach().float()


def _require_finite_values(values: list[Tensor], label: str) -> None:
    if values and not bool(
        torch.stack([torch.isfinite(value).all() for value in values]).all()
    ):
        raise BenchmarkExecutionError(f"non-finite {label}")
