from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tinyllm.config.schema import ModelConfig
from tinyllm.training.benchmark import (
    BenchmarkExecutionError,
    BenchmarkRuntime,
    PrecisionBenchmarkConfig,
    benchmark_precision,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        mlp_ratio=2.0,
        dropout=0.0,
        rope_theta=10_000.0,
        rms_norm_eps=1.0e-5,
    )


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.seen: list[torch.Tensor] = []

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor) -> object:
        self.seen.append(input_ids.detach().cpu().clone())
        target = input_ids.float().mean() / 16.0
        return SimpleNamespace(loss=(self.weight - target).square() + 0.1)


class LinearLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(16, 16, bias=False)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor) -> object:
        activations = torch.nn.functional.one_hot(input_ids, num_classes=16).float()
        return SimpleNamespace(loss=self.projection(activations).square().mean())


class DeviceAgnosticLinearLossModel(LinearLossModel):
    def to(self, *args, **kwargs):
        return self


class FakeDeviceMetrics:
    def __init__(self, events: list[str] | None = None) -> None:
        self.reset_devices: list[torch.device] = []
        self.synchronized_devices: list[torch.device] = []
        self.cleared_devices: list[torch.device] = []
        self.events = [] if events is None else events

    def clear_cache(self, device: torch.device) -> None:
        self.cleared_devices.append(device)
        self.events.append("clear")

    def reset_peak_memory_stats(self, device: torch.device) -> None:
        self.reset_devices.append(device)
        self.events.append("reset")

    def synchronize(self, device: torch.device) -> None:
        self.synchronized_devices.append(device)
        self.events.append("sync")

    def peak_allocated_bytes(self, device: torch.device) -> int:
        return 111

    def peak_reserved_bytes(self, device: torch.device) -> int:
        return 222


class FakePolicy:
    name = "fp32"

    def autocast(self):
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()


def _optimizer(model: nn.Module, learning_rate: float, weight_decay: float):
    return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def test_benchmark_reuses_identical_batches_and_emits_machine_readable_results(
    tmp_path: Path,
) -> None:
    models: list[RecordingModel] = []
    metrics = FakeDeviceMetrics()
    ticks = iter((10.0, 12.0, 20.0, 22.0))

    def model_factory(config: ModelConfig) -> RecordingModel:
        model = RecordingModel()
        models.append(model)
        return model

    output = tmp_path / "precision.json"
    config = PrecisionBenchmarkConfig(
        model=_model_config(),
        device="cpu",
        batch_size=2,
        sequence_length=4,
        warmup_steps=1,
        measured_steps=2,
        seed=1337,
        learning_rate=0.01,
        weight_decay=0.0,
        compile=False,
        output_path=output,
    )
    runtime = BenchmarkRuntime(
        model_factory=model_factory,
        optimizer_factory=_optimizer,
        timer=lambda: next(ticks),
        device_metrics=metrics,
    )

    results = benchmark_precision(config, ["fp32", "bf16"], runtime=runtime)

    measured_models = [model for model in models if model.seen]
    assert len(measured_models) == 2
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            measured_models[0].seen,
            measured_models[1].seen,
            strict=True,
        )
    )
    assert [result.mode for result in results] == ["fp32", "bf16"]
    assert all(result.status == "ready" for result in results)
    assert all(result.reason is None for result in results)
    assert all(result.seed == 1337 for result in results)
    assert all(result.batch_shape == (2, 4) for result in results)
    assert all(result.tokens_per_second == pytest.approx(8.0) for result in results)
    assert all(result.peak_allocated_bytes == 111 for result in results)
    assert all(result.peak_reserved_bytes == 222 for result in results)
    assert all(result.finite_loss for result in results)
    assert all(result.mean_loss is not None for result in results)
    assert len(metrics.reset_devices) == 2
    assert len(metrics.synchronized_devices) == 4
    assert len(metrics.cleared_devices) == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format_version"] == 2
    assert payload["results"] == [result.to_dict() for result in results]


def test_benchmark_records_declared_failure_and_continues_remaining_modes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "precision.json"
    metrics = FakeDeviceMetrics()
    ticks = iter((1.0, 2.0))
    config = PrecisionBenchmarkConfig(
        model=_model_config(),
        device="cpu",
        batch_size=1,
        sequence_length=2,
        warmup_steps=0,
        measured_steps=1,
        seed=7,
        learning_rate=0.01,
        weight_decay=0.0,
        compile=False,
        output_path=output,
    )

    def policy_factory(mode: str, device: torch.device):
        if mode == "bf16":
            raise BenchmarkExecutionError("kernel exploded")
        return FakePolicy()

    runtime = BenchmarkRuntime(
        model_factory=lambda config: RecordingModel(),
        optimizer_factory=_optimizer,
        policy_factory=policy_factory,
        timer=lambda: next(ticks),
        device_metrics=metrics,
    )

    results = benchmark_precision(config, ["bf16", "fp32"], runtime=runtime)

    assert [result.status for result in results] == ["failed", "ready"]
    assert results[0].reason == "BenchmarkExecutionError: kernel exploded"
    assert results[1].tokens_per_second == pytest.approx(2.0)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["results"] == [result.to_dict() for result in results]


def test_benchmark_streams_device_batches_after_reset_and_reads_losses_once() -> None:
    events: list[str] = []
    metrics = FakeDeviceMetrics(events)
    ticks = iter((1.0, 2.0))
    read_buffers: list[torch.Tensor] = []

    def transfer(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs.device.type == "cpu"
        assert targets.device.type == "cpu"
        events.append("transfer")
        return inputs.to(device), targets.to(device)

    def read_losses(losses: torch.Tensor) -> list[float]:
        events.append("read-losses")
        read_buffers.append(losses)
        return losses.cpu().tolist()

    runtime = BenchmarkRuntime(
        model_factory=lambda config: RecordingModel(),
        optimizer_factory=_optimizer,
        timer=lambda: next(ticks),
        device_metrics=metrics,
        batch_to_device=transfer,
        loss_reader=read_losses,
    )
    config = PrecisionBenchmarkConfig(
        model=_model_config(),
        device="cpu",
        batch_size=1,
        sequence_length=2,
        warmup_steps=1,
        measured_steps=2,
        seed=7,
        learning_rate=0.01,
        weight_decay=0.0,
        compile=False,
    )

    benchmark_precision(config, ["fp32"], runtime=runtime)

    assert events == [
        "transfer",
        "clear",
        "reset",
        "sync",
        "transfer",
        "transfer",
        "sync",
        "read-losses",
    ]
    assert len(read_buffers) == 1
    assert read_buffers[0].shape == (2,)


@pytest.mark.parametrize("modes", [[], ["fp32", "fp32"], ["automatic"]])
def test_benchmark_rejects_invalid_mode_lists(modes: list[str]) -> None:
    config = PrecisionBenchmarkConfig(
        model=_model_config(),
        device="cpu",
        batch_size=1,
        sequence_length=2,
        warmup_steps=0,
        measured_steps=1,
        seed=7,
        learning_rate=0.01,
        weight_decay=0.0,
        compile=False,
    )

    with pytest.raises(ValueError, match="modes"):
        benchmark_precision(config, modes)


class NoOpPolicy(FakePolicy):
    def backward(self, loss: torch.Tensor) -> None:
        del loss

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer


def _one_step_benchmark(tmp_path: Path | None = None) -> PrecisionBenchmarkConfig:
    return PrecisionBenchmarkConfig(
        model=_model_config(),
        device="cpu",
        batch_size=1,
        sequence_length=2,
        warmup_steps=0,
        measured_steps=1,
        seed=7,
        learning_rate=0.01,
        weight_decay=0.0,
        compile=False,
        output_path=None if tmp_path is None else tmp_path / "precision.json",
    )


def test_benchmark_non_finite_loss_is_failed_not_ready(tmp_path: Path) -> None:
    class NonFiniteLossModel(RecordingModel):
        def forward(self, input_ids: torch.Tensor, targets: torch.Tensor) -> object:
            del input_ids, targets
            return SimpleNamespace(loss=self.weight * float("nan"))

    result = benchmark_precision(
        _one_step_benchmark(tmp_path),
        ["fp32"],
        runtime=BenchmarkRuntime(
            model_factory=lambda config: NonFiniteLossModel(),
            optimizer_factory=_optimizer,
            policy_factory=lambda mode, device: NoOpPolicy(),
            timer=iter((1.0, 2.0)).__next__,
            device_metrics=FakeDeviceMetrics(),
        ),
    )[0]

    assert result.status == "failed"
    assert result.finite_loss is False
    assert result.reason == "non-finite measured loss"
    payload = json.loads((_one_step_benchmark(tmp_path).output_path).read_text())
    assert payload["results"][0]["status"] == "failed"


@pytest.mark.parametrize("failure", ["gradient", "parameter"])
def test_benchmark_checks_finite_optimizer_state_after_step(failure: str) -> None:
    class CorruptingPolicy(FakePolicy):
        def backward(self, loss: torch.Tensor) -> None:
            super().backward(loss)
            if failure == "gradient":
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.fill_(float("inf"))

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            super().step(optimizer)
            if failure == "parameter":
                with torch.no_grad():
                    next(model.parameters()).fill_(float("nan"))

    model = RecordingModel()
    result = benchmark_precision(
        _one_step_benchmark(),
        ["fp32"],
        runtime=BenchmarkRuntime(
            model_factory=lambda config: model,
            optimizer_factory=_optimizer,
            policy_factory=lambda mode, device: CorruptingPolicy(),
            timer=iter((1.0, 2.0)).__next__,
            device_metrics=FakeDeviceMetrics(),
        ),
    )[0]

    assert result.status == "failed"
    assert f"non-finite {failure}" in result.reason


def test_benchmark_does_not_swallow_generic_runtime_programming_error() -> None:
    unexpected = RuntimeError("programming bug")

    def policy_factory(mode: str, device: torch.device):
        del mode, device
        raise unexpected

    with pytest.raises(RuntimeError) as raised:
        benchmark_precision(
            _one_step_benchmark(),
            ["fp32"],
            runtime=BenchmarkRuntime(
                model_factory=lambda config: RecordingModel(),
                optimizer_factory=_optimizer,
                policy_factory=policy_factory,
                device_metrics=FakeDeviceMetrics(),
            ),
        )

    assert raised.value is unexpected
