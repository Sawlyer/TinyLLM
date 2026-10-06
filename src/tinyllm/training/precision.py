from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from tinyllm.errors import PrecisionUnavailableError
from tinyllm.training import capabilities


@dataclass(slots=True)
class PrecisionPolicy:
    """Explicit autocast and gradient-scaling behavior for one precision mode."""

    name: str
    device: torch.device
    autocast_dtype: torch.dtype | None
    scaler: torch.amp.GradScaler | None
    fp8_capability: capabilities.FP8Capability | None = None

    @classmethod
    def create(
        cls,
        name: str,
        device: torch.device | str,
        *,
        fp8_capability: capabilities.FP8Capability | None = None,
    ) -> PrecisionPolicy:
        if not isinstance(name, str):
            raise TypeError("precision name must be a string")
        normalized = name.lower()
        resolved_device = torch.device(device)
        if normalized == "fp32":
            return cls(normalized, resolved_device, None, None)
        if normalized == "bf16":
            if resolved_device.type not in {"cpu", "cuda"}:
                raise RuntimeError("BF16 autocast requires a CPU or CUDA device")
            if resolved_device.type == "cuda" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 is unsupported by this CUDA device; use fp16 or fp32")
            return cls(normalized, resolved_device, torch.bfloat16, None)
        if normalized == "fp16":
            if resolved_device.type not in {"cpu", "cuda"}:
                raise RuntimeError("FP16 autocast requires a CPU or CUDA device")
            if resolved_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("FP16 CUDA precision requested but CUDA is unavailable")
            scaler = torch.amp.GradScaler(resolved_device.type, enabled=True)
            return cls(normalized, resolved_device, torch.float16, scaler)
        if normalized == "fp8":
            capability = fp8_capability
            if capability is None:
                capability = capabilities.probe_fp8_capability(resolved_device)
            if not capabilities.supports_fp8(resolved_device, capability=capability):
                reason = capabilities.fp8_unavailable_reason(
                    resolved_device,
                    capability=capability,
                )
                detail = reason or f"device {resolved_device} failed capability probe"
                raise PrecisionUnavailableError(
                    "FP8 requires TorchAO, supported CUDA hardware with compute capability 8.9+, "
                    f"and torch.compile; capability unavailable: {detail}. "
                    "Install tinyllm[fp8] with a compatible CUDA PyTorch build"
                )
            return cls(normalized, resolved_device, torch.bfloat16, None, capability)
        raise ValueError("precision must be one of bf16, fp16, fp32, fp8")

    @property
    def requires_compile(self) -> bool:
        return self.name == "fp8"

    def prepare_model(
        self,
        model: nn.Module,
        *,
        float8_backend: capabilities.Float8Backend | None = None,
    ) -> nn.Module:
        if self.name != "fp8":
            return model
        return capabilities.convert_linear_layers_to_fp8(
            model,
            backend=float8_backend,
            capability=self.fp8_capability,
        )

    def autocast(self) -> AbstractContextManager[object]:
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def backward(self, loss: Tensor) -> None:
        if self.scaler is None:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def step(self, optimizer: Optimizer) -> None:
        if self.scaler is None:
            optimizer.step()
        else:
            self.scaler.step(optimizer)
            self.scaler.update()

    def state_dict(self) -> dict[str, object]:
        return {} if self.scaler is None else self.scaler.state_dict()

    def load_state_dict(self, state: dict[str, object]) -> None:
        if self.scaler is None:
            if state:
                raise ValueError(f"{self.name} precision does not use gradient scaling")
            return
        self.scaler.load_state_dict(state)
