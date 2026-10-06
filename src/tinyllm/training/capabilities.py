"""Runtime capability probes for optional precision features."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from tinyllm.errors import OptionalBackendUnavailableError, PrecisionUnavailableError


class Float8Backend(Protocol):
    """Minimal TorchAO surface used for Float8Linear conversion."""

    def convert(
        self,
        model: nn.Module,
        module_filter_fn: Callable[[nn.Module, str], bool],
    ) -> None: ...

    def is_float8_linear(self, module: nn.Module) -> bool: ...


class _TorchAOFloat8Backend:
    def __init__(self, float8_module: object, float8_linear_module: object) -> None:
        self._config_type = getattr(float8_module, "Float8LinearConfig")
        self._converter = getattr(float8_module, "convert_to_float8_training")
        self._linear_type = getattr(float8_linear_module, "Float8Linear")

    def convert(
        self,
        model: nn.Module,
        module_filter_fn: Callable[[nn.Module, str], bool],
    ) -> None:
        config = self._config_type.from_recipe_name("tensorwise")
        self._converter(model, config=config, module_filter_fn=module_filter_fn)

    def is_float8_linear(self, module: nn.Module) -> bool:
        return isinstance(module, self._linear_type)


@dataclass(frozen=True, slots=True)
class FP8Capability:
    """One reusable kernel-probe result bound to a requested device context."""

    device: torch.device
    unavailable_reason: str | None

    @property
    def supported(self) -> bool:
        return self.unavailable_reason is None


def probe_fp8_capability(device: torch.device | str = "cuda") -> FP8Capability:
    """Evaluate FP8 prerequisites once and return a reusable result."""
    resolved_device = torch.device(device)
    return FP8Capability(
        device=resolved_device,
        unavailable_reason=_evaluate_fp8_unavailable_reason(resolved_device),
    )


def fp8_unavailable_reason(
    device: torch.device | str = "cuda",
    *,
    capability: FP8Capability | None = None,
) -> str | None:
    """Return exact missing FP8 prerequisite from one device-bound result."""
    return _resolve_fp8_capability(device, capability).unavailable_reason


def _evaluate_fp8_unavailable_reason(resolved_device: torch.device) -> str | None:
    if resolved_device.type != "cuda":
        return f"FP8 requires a CUDA device, got {resolved_device.type}"
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    if resolved_device.index is not None and resolved_device.index >= torch.cuda.device_count():
        return f"CUDA device index {resolved_device.index} is unavailable"
    try:
        compute_capability = torch.cuda.get_device_capability(resolved_device)
    except AssertionError as error:
        return f"CUDA device capability probe failed: {error}"
    if compute_capability < (8, 9):
        rendered = ".".join(str(part) for part in compute_capability)
        return f"CUDA compute capability {rendered} is below required 8.9"
    if not hasattr(torch, "compile"):
        return "torch.compile is unavailable"
    return _probe_fp8_kernel(resolved_device)


def supports_fp8(
    device: torch.device | str = "cuda",
    *,
    capability: FP8Capability | None = None,
) -> bool:
    """Report whether required TorchAO, CUDA, hardware, and compiler APIs exist."""
    return _resolve_fp8_capability(device, capability).supported


def require_fp8(
    device: torch.device | str = "cuda",
    *,
    capability: FP8Capability | None = None,
) -> None:
    """Raise actionable error instead of selecting a lower precision silently."""
    reason = _resolve_fp8_capability(device, capability).unavailable_reason
    if reason is not None:
        raise PrecisionUnavailableError(_fp8_error(reason))


def convert_linear_layers_to_fp8(
    model: nn.Module,
    *,
    backend: Float8Backend | None = None,
    capability: FP8Capability | None = None,
) -> nn.Module:
    """Convert eligible ``nn.Linear`` modules with TorchAO, in place."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    model_device = _model_device(model)
    if capability is not None:
        _resolve_fp8_capability(model_device, capability)
    resolved_backend = backend
    if resolved_backend is None:
        require_fp8(model_device, capability=capability)
        resolved_backend = _load_torchao_float8_backend()
    eligible = [
        name for name, module in model.named_modules() if _eligible_float8_linear(module, name)
    ]
    if not eligible:
        raise PrecisionUnavailableError(
            "FP8 TorchAO conversion found no eligible Linear layers; "
            "input and output dimensions must be divisible by 16"
        )
    resolved_backend.convert(model, _eligible_float8_linear)
    converted = [
        name for name in eligible if resolved_backend.is_float8_linear(model.get_submodule(name))
    ]
    if converted != eligible:
        missing = sorted(set(eligible) - set(converted))
        raise PrecisionUnavailableError(
            f"FP8 TorchAO conversion left eligible Linear layers unconverted: {', '.join(missing)}"
        )
    return model


def _eligible_float8_linear(module: nn.Module, qualified_name: str) -> bool:
    del qualified_name
    return (
        isinstance(module, nn.Linear)
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
    )


def _load_torchao_float8_backend() -> Float8Backend:
    try:
        float8_module = importlib.import_module("torchao.float8")
        float8_linear_module = importlib.import_module("torchao.float8.float8_linear")
        getattr(float8_module, "Float8LinearConfig")
        getattr(float8_module, "convert_to_float8_training")
        getattr(float8_linear_module, "Float8Linear")
    except (ImportError, AttributeError) as error:
        raise OptionalBackendUnavailableError(
            "TorchAO float8 conversion APIs or converted Linear type are unavailable"
        ) from error
    return _TorchAOFloat8Backend(float8_module, float8_linear_module)


def _probe_fp8_kernel(device: torch.device) -> str | None:
    try:
        backend = _load_torchao_float8_backend()
        model = nn.Sequential(nn.Linear(16, 16, bias=False, device=device, dtype=torch.bfloat16))
        backend.convert(model, _eligible_float8_linear)
        if not backend.is_float8_linear(model[0]):
            return "TorchAO FP8 probe did not create Float8Linear"
        compiled = torch.compile(model)
        inputs = torch.randn(16, 16, device=device, dtype=torch.bfloat16)
        loss = compiled(inputs).square().mean()
        loss.backward()
        torch.cuda.synchronize(device)
        if not bool(torch.isfinite(loss)):
            return "Float8Linear compiled kernel produced non-finite loss"
    except OptionalBackendUnavailableError as error:
        return _render_kernel_probe_error(device, error)
    except (ImportError, NotImplementedError) as error:
        return _render_kernel_probe_error(device, error)
    except Exception as error:
        if not _is_known_optional_kernel_error(error):
            raise
        return _render_kernel_probe_error(device, error)
    return None


def _render_kernel_probe_error(device: torch.device, error: BaseException) -> str:
    return (
        f"Float8Linear compiled kernel probe failed on {device}: "
        f"{type(error).__name__}: {error}"
    )


def _is_known_optional_kernel_error(error: BaseException) -> bool:
    known_prefixes = ("torchao", "torch._dynamo", "torch._inductor", "triton")
    if type(error).__module__.startswith(known_prefixes):
        return True
    traceback = error.__traceback__
    while traceback is not None:
        module_name = str(traceback.tb_frame.f_globals.get("__name__", ""))
        if module_name.startswith(known_prefixes):
            return True
        traceback = traceback.tb_next
    return False


def _model_device(model: nn.Module) -> torch.device:
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1:
        raise PrecisionUnavailableError("FP8 model parameters must be on one device")
    return next(iter(devices))


def _resolve_fp8_capability(
    device: torch.device | str,
    capability: FP8Capability | None,
) -> FP8Capability:
    resolved_device = torch.device(device)
    if capability is None:
        return probe_fp8_capability(resolved_device)
    if not _devices_share_context(resolved_device, capability.device):
        raise ValueError(
            f"FP8 capability for {capability.device} cannot be used with {resolved_device}"
        )
    return capability


def _devices_share_context(left: torch.device, right: torch.device) -> bool:
    if left.type != right.type:
        return False
    return left.index is None or right.index is None or left.index == right.index


def _fp8_error(reason: str) -> str:
    return (
        "FP8 requires TorchAO, supported CUDA hardware with compute capability 8.9+, "
        f"and torch.compile; capability unavailable: {reason}. Install tinyllm[fp8] "
        "with a compatible CUDA PyTorch build"
    )
