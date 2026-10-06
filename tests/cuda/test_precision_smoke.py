from __future__ import annotations

import pytest
import torch
from torch import nn

from tinyllm.training import capabilities
from tinyllm.training.benchmark import convert_linear_layers_to_fp8
from tinyllm.training.precision import PrecisionPolicy

pytestmark = pytest.mark.cuda


def test_bf16_cuda_forward_backward_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = PrecisionPolicy.create("bf16", torch.device("cuda"))
    model = nn.Linear(32, 32, bias=False).cuda()
    inputs = torch.randn(4, 32, device="cuda")

    with policy.autocast():
        loss = model(inputs).square().mean()
    policy.backward(loss)

    assert torch.isfinite(loss)


def test_fp8_cuda_conversion_compile_forward_backward_smoke() -> None:
    device = torch.device("cuda")
    capability = capabilities.probe_fp8_capability(device)
    reason = capabilities.fp8_unavailable_reason(device, capability=capability)
    if reason is not None:
        pytest.skip(f"FP8 unavailable: {reason}")
    policy = PrecisionPolicy.create("fp8", device, fp8_capability=capability)
    model = nn.Sequential(
        nn.Linear(32, 32, bias=False),
        nn.GELU(),
        nn.Linear(32, 32, bias=False),
    ).cuda()
    model = convert_linear_layers_to_fp8(model, capability=capability)

    converted = [
        module
        for module in model.modules()
        if type(module).__module__ == "torchao.float8.float8_linear"
        and type(module).__name__ == "Float8Linear"
    ]
    assert len(converted) == 2
    compiled = torch.compile(model)
    inputs = torch.randn(4, 32, device="cuda")

    with policy.autocast():
        loss = compiled(inputs).square().mean()
    policy.backward(loss)

    assert torch.isfinite(loss)
