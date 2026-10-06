from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tinyllm.training import capabilities
from tinyllm.training.precision import PrecisionPolicy


class FakeFloat8Linear(nn.Linear):
    pass


class HonestFakeFloat8Backend:
    def convert(self, model: nn.Module, module_filter_fn) -> None:
        for name, module in tuple(model.named_modules()):
            if not module_filter_fn(module, name):
                continue
            replacement = FakeFloat8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            replacement.load_state_dict(module.state_dict())
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, replacement)

    def is_float8_linear(self, module: nn.Module) -> bool:
        return isinstance(module, FakeFloat8Linear)


def test_torchao_backend_uses_internal_converted_type_not_removed_public_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConfig:
        @classmethod
        def from_recipe_name(cls, name: str) -> object:
            assert name == "tensorwise"
            return object()

    public_module = SimpleNamespace(
        Float8LinearConfig=FakeConfig,
        convert_to_float8_training=lambda model, **kwargs: model,
    )
    internal_module = SimpleNamespace(Float8Linear=FakeFloat8Linear)
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return {
            "torchao.float8": public_module,
            "torchao.float8.float8_linear": internal_module,
        }[name]

    monkeypatch.setattr(capabilities.importlib, "import_module", import_module)

    backend = capabilities._load_torchao_float8_backend()

    assert imported == ["torchao.float8", "torchao.float8.float8_linear"]
    assert backend.is_float8_linear(FakeFloat8Linear(16, 16))


def test_fp32_and_bf16_do_not_create_gradient_scalers() -> None:
    fp32 = PrecisionPolicy.create("fp32", torch.device("cpu"))
    bf16 = PrecisionPolicy.create("bf16", torch.device("cpu"))

    assert fp32.autocast_dtype is None
    assert fp32.scaler is None
    assert bf16.autocast_dtype is torch.bfloat16
    assert bf16.scaler is None


def test_fp16_owns_an_enabled_gradient_scaler() -> None:
    policy = PrecisionPolicy.create("fp16", torch.device("cpu"))

    assert policy.autocast_dtype is torch.float16
    assert policy.scaler is not None
    assert policy.scaler.is_enabled()


def test_fp16_rejects_unavailable_cuda_instead_of_disabling_scaler(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="FP16.*CUDA"):
        PrecisionPolicy.create("fp16", torch.device("cuda"))


def test_fp8_never_silently_falls_back_without_torchao_and_cuda() -> None:
    with pytest.raises(RuntimeError, match="FP8.*TorchAO.*CUDA"):
        PrecisionPolicy.create("fp8", torch.device("cpu"))


def test_fp8_never_silently_falls_back_when_capability_probe_fails(monkeypatch) -> None:
    devices: list[torch.device] = []
    monkeypatch.setattr(
        capabilities,
        "probe_fp8_capability",
        lambda device: (
            devices.append(torch.device(device))
            or capabilities.FP8Capability(
                torch.device(device),
                "CUDA capability unavailable",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="FP8.*TorchAO.*CUDA.*torch.compile"):
        PrecisionPolicy.create("fp8", torch.device("cuda:1"))

    assert devices == [torch.device("cuda:1")]


def test_fp8_policy_uses_bf16_for_non_linear_operations(monkeypatch) -> None:
    monkeypatch.setattr(
        capabilities,
        "probe_fp8_capability",
        lambda device: capabilities.FP8Capability(torch.device(device), None),
    )

    policy = PrecisionPolicy.create("fp8", torch.device("cuda"))

    assert policy.name == "fp8"
    assert policy.autocast_dtype is torch.bfloat16
    assert policy.scaler is None


def test_fp8_capability_probes_requested_device_and_minimal_kernel(monkeypatch) -> None:
    calls: list[tuple[str, torch.device]] = []
    requested = torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: calls.append(("capability", torch.device(device))) or (12, 0),
    )
    monkeypatch.setattr(
        capabilities,
        "_probe_fp8_kernel",
        lambda device: calls.append(("kernel", torch.device(device))) or None,
    )

    assert capabilities.supports_fp8(requested)
    assert calls == [("capability", requested), ("kernel", requested)]


def test_fp8_capability_rejects_non_cuda_device_before_probe() -> None:
    assert capabilities.fp8_unavailable_reason(torch.device("cpu")) == (
        "FP8 requires a CUDA device, got cpu"
    )


def test_fp8_capability_reports_kernel_probe_failure(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        capabilities,
        "_probe_fp8_kernel",
        lambda device: "Float8Linear compiled GEMM failed: missing kernel",
    )

    assert capabilities.fp8_unavailable_reason(torch.device("cuda:0")) == (
        "Float8Linear compiled GEMM failed: missing kernel"
    )


def test_fp8_success_context_probes_kernel_once_through_policy_and_conversion(
    monkeypatch,
) -> None:
    requested = torch.device("cuda:0")
    probe_devices: list[torch.device] = []
    backend = HonestFakeFloat8Backend()
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

    policy = PrecisionPolicy.create("fp8", requested)
    assert capabilities.supports_fp8(requested, capability=policy.fp8_capability)
    assert capabilities.fp8_unavailable_reason(requested, capability=policy.fp8_capability) is None
    capabilities.require_fp8(requested, capability=policy.fp8_capability)
    converted = policy.prepare_model(nn.Sequential(nn.Linear(16, 16)))

    assert isinstance(converted[0], FakeFloat8Linear)
    assert probe_devices == [requested]


def test_fp8_conversion_rejects_injected_backend_for_different_capability_device(
    monkeypatch,
) -> None:
    backend = HonestFakeFloat8Backend()
    capability = capabilities.FP8Capability(torch.device("cuda:0"), None)
    monkeypatch.setattr(
        capabilities,
        "_model_device",
        lambda model: torch.device("cuda:1"),
    )
    monkeypatch.setattr(
        capabilities,
        "probe_fp8_capability",
        lambda device: pytest.fail("conversion must reuse supplied capability"),
    )

    with pytest.raises(ValueError, match="cuda:0.*cuda:1"):
        capabilities.convert_linear_layers_to_fp8(
            nn.Sequential(nn.Linear(16, 16)),
            backend=backend,
            capability=capability,
        )


def test_fp8_failure_context_probes_kernel_once_and_reuses_exact_reason(monkeypatch) -> None:
    requested = torch.device("cuda:0")
    probe_devices: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(
        capabilities,
        "_probe_fp8_kernel",
        lambda device: (
            probe_devices.append(torch.device(device))
            or "Float8Linear compiled GEMM failed: missing kernel"
        ),
    )

    with pytest.raises(RuntimeError, match="missing kernel"):
        PrecisionPolicy.create("fp8", requested)

    assert probe_devices == [requested]


def test_fp8_policy_includes_exact_capability_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        capabilities,
        "probe_fp8_capability",
        lambda device: capabilities.FP8Capability(
            torch.device(device),
            "Float8Linear compiled GEMM failed: missing kernel",
        ),
    )

    with pytest.raises(RuntimeError, match="missing kernel"):
        PrecisionPolicy.create("fp8", torch.device("cuda:0"))


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision"):
        PrecisionPolicy.create("automatic", torch.device("cpu"))


def test_fp8_kernel_probe_does_not_swallow_generic_runtime_error(monkeypatch) -> None:
    unexpected = RuntimeError("programming bug")

    def broken_backend():
        raise unexpected

    monkeypatch.setattr(capabilities, "_load_torchao_float8_backend", broken_backend)

    with pytest.raises(RuntimeError) as raised:
        capabilities._probe_fp8_kernel(torch.device("cuda"))

    assert raised.value is unexpected
