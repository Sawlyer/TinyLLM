from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tinyllm.config.schema import ModelConfig
from tinyllm.training import capabilities
from tinyllm.training.benchmark import (
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


class FakeFP8Policy:
    name = "fp8"

    def __init__(self, capability=None) -> None:
        self.fp8_capability = capability

    def autocast(self):
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()


class FakeFloat8Linear(nn.Linear):
    pass


class FakeFloat8Backend:
    def __init__(self) -> None:
        self.eligible_names: list[str] = []

    def convert(self, model: nn.Module, module_filter_fn) -> None:
        self.eligible_names = [
            name for name, module in model.named_modules() if module_filter_fn(module, name)
        ]
        for name in self.eligible_names:
            source = model.get_submodule(name)
            replacement = FakeFloat8Linear(
                source.in_features,
                source.out_features,
                bias=source.bias is not None,
            )
            replacement.load_state_dict(source.state_dict())
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, replacement)

    def is_float8_linear(self, module: nn.Module) -> bool:
        return isinstance(module, FakeFloat8Linear)


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


def test_benchmark_persists_ready_mode_when_another_mode_is_unsupported(
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
    runtime = BenchmarkRuntime(
        model_factory=lambda config: RecordingModel(),
        optimizer_factory=_optimizer,
        timer=lambda: next(ticks),
        device_metrics=metrics,
    )

    results = benchmark_precision(config, ["fp32", "fp8"], runtime=runtime)

    assert [result.status for result in results] == ["ready", "unsupported"]
    assert results[0].tokens_per_second == pytest.approx(2.0)
    assert results[1].tokens_per_second is None
    assert results[1].reason is not None
    assert "FP8 requires a CUDA device" in results[1].reason
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["results"] == [result.to_dict() for result in results]


def test_benchmark_records_unexpected_failure_and_continues_remaining_modes(
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
            raise RuntimeError("kernel exploded")
        return FakeFP8Policy()

    runtime = BenchmarkRuntime(
        model_factory=lambda config: RecordingModel(),
        optimizer_factory=_optimizer,
        policy_factory=policy_factory,
        timer=lambda: next(ticks),
        device_metrics=metrics,
    )

    results = benchmark_precision(config, ["bf16", "fp32"], runtime=runtime)

    assert [result.status for result in results] == ["failed", "ready"]
    assert results[0].reason == "RuntimeError: kernel exploded"
    assert results[1].tokens_per_second == pytest.approx(2.0)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["results"] == [result.to_dict() for result in results]


def test_fp8_benchmark_converts_eligible_linear_layers_and_always_compiles() -> None:
    backend = FakeFloat8Backend()
    compiled: list[nn.Module] = []
    metrics = FakeDeviceMetrics()
    ticks = iter((1.0, 2.0))

    def compiler(model: nn.Module) -> nn.Module:
        compiled.append(model)
        return model

    runtime = BenchmarkRuntime(
        model_factory=lambda config: LinearLossModel(),
        optimizer_factory=_optimizer,
        policy_factory=lambda mode, device: FakeFP8Policy(),
        compiler=compiler,
        float8_backend=backend,
        timer=lambda: next(ticks),
        device_metrics=metrics,
    )
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

    result = benchmark_precision(config, ["fp8"], runtime=runtime)[0]

    assert backend.eligible_names == ["projection"]
    assert len(compiled) == 1
    assert result.mode == "fp8"


def test_fp8_benchmark_reuses_policy_capability_without_second_kernel_probe(
    monkeypatch,
) -> None:
    requested = torch.device("cuda:0")
    probe_devices: list[torch.device] = []
    backend = FakeFloat8Backend()
    metrics = FakeDeviceMetrics()
    ticks = iter((1.0, 2.0))
    torch_empty = torch.empty

    def device_agnostic_empty(*args, **kwargs):
        requested_device = kwargs.get("device")
        if requested_device is not None and torch.device(requested_device).type == "cuda":
            kwargs["device"] = "cpu"
        return torch_empty(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        capabilities,
        "_probe_fp8_kernel",
        lambda device: probe_devices.append(torch.device(device)) or None,
    )
    monkeypatch.setattr(capabilities, "_model_device", lambda model: requested)
    monkeypatch.setattr(capabilities, "_load_torchao_float8_backend", lambda: backend)
    monkeypatch.setattr(torch, "empty", device_agnostic_empty)
    capability = capabilities.probe_fp8_capability(requested)
    runtime = BenchmarkRuntime(
        model_factory=lambda config: DeviceAgnosticLinearLossModel(),
        optimizer_factory=_optimizer,
        policy_factory=lambda mode, device: FakeFP8Policy(capability),
        compiler=lambda model: model,
        timer=lambda: next(ticks),
        device_metrics=metrics,
        batch_to_device=lambda inputs, targets, device: (inputs, targets),
    )
    config = PrecisionBenchmarkConfig(
        model=_model_config(),
        device=requested,
        batch_size=1,
        sequence_length=2,
        warmup_steps=0,
        measured_steps=1,
        seed=7,
        learning_rate=0.01,
        weight_decay=0.0,
        compile=False,
    )

    benchmark_precision(config, ["fp8"], runtime=runtime)

    assert backend.eligible_names == ["projection"]
    assert probe_devices == [requested]


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
