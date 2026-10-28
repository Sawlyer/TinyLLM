import json
from types import SimpleNamespace

import pytest
import torch

from tinyllm.config.schema import ModelConfig
from tinyllm.inference.memory import (
    estimate_kv_cache_memory,
    require_available_kv_cache_memory,
)


def make_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        max_seq_len=32,
        d_model=8,
        n_layers=3,
        n_heads=2,
        n_kv_heads=1,
        mlp_ratio=2,
        dropout=0.0,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )


def test_kv_cache_estimate_uses_layers_batch_context_and_dtype():
    estimate = estimate_kv_cache_memory(
        make_model_config(),
        batch_size=2,
        context_length=10,
        dtype=torch.bfloat16,
    )

    assert estimate.bytes_per_layer == 320
    assert estimate.total_bytes == 960
    assert estimate.to_dict()["dtype"] == "bfloat16"


@pytest.mark.parametrize(
    ("batch_size", "context_length", "dtype"),
    [(0, 10, torch.float32), (1, 0, torch.float32), (1, 33, torch.float32), (1, 10, torch.int64)],
)
def test_kv_cache_estimate_rejects_invalid_dimensions_and_dtype(
    batch_size, context_length, dtype
):
    with pytest.raises(ValueError):
        estimate_kv_cache_memory(
            make_model_config(),
            batch_size=batch_size,
            context_length=context_length,
            dtype=dtype,
        )


def test_kv_cache_preflight_explains_insufficient_available_memory():
    estimate = estimate_kv_cache_memory(
        make_model_config(),
        batch_size=1,
        context_length=32,
        dtype=torch.float32,
    )

    with pytest.raises(MemoryError, match="Reduce model.max_seq_len"):
        require_available_kv_cache_memory(estimate, available_bytes=0)


def test_estimate_kv_cache_command_emits_json(monkeypatch, capsys):
    from tinyllm import cli

    monkeypatch.setattr(
        cli,
        "_load_args_config",
        lambda _args: SimpleNamespace(model=make_model_config()),
    )
    cli._estimate_kv_cache_command(
        SimpleNamespace(
            config=None,
            overrides=[],
            batch_size=2,
            context_length=None,
            dtype="float32",
        )
    )

    report = json.loads(capsys.readouterr().out)
    assert report["batch_size"] == 2
    assert report["context_length"] == 32
    assert report["total_bytes"] == 3 * 2 * 2 * 1 * 4 * 32 * 4


def test_estimate_kv_cache_parser_accepts_configuration_and_dtype():
    from tinyllm import cli

    args = cli._build_parser().parse_args(
        [
            "estimate-kv-cache",
            "--config",
            "configs/tinyllm.yaml",
            "--context-length",
            "16",
            "--batch-size",
            "2",
            "--dtype",
            "bfloat16",
        ]
    )

    assert args.context_length == 16
    assert args.batch_size == 2
    assert args.dtype == "bfloat16"
    assert args.handler.__name__ == "_estimate_kv_cache_command"
