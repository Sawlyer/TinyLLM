from __future__ import annotations

import pytest
import torch
from torch import nn

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
