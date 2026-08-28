import pytest
import torch

from tinyllm.training.precision import PrecisionPolicy


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


def test_unknown_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision"):
        PrecisionPolicy.create("automatic", torch.device("cpu"))
