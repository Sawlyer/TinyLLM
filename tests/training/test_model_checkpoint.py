from pathlib import Path

import pytest
import torch

from tinyllm.config.schema import ModelConfig
from tinyllm.errors import CheckpointFormatError
from tinyllm.model.transformer import TinyLLM
from tinyllm.training.checkpoint import (
    ArtifactIdentity,
    load_checkpoint,
    load_model_checkpoint,
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


def test_model_only_loader_ignores_training_state_and_cuda_topology(
    tmp_path: Path,
) -> None:
    config = _model_config()
    model = TinyLLM(config)
    checkpoint = tmp_path / "cuda-training.pt"
    identity = {
        "model_config": config.model_dump(mode="json"),
        "tokenizer_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
    }
    torch.save(
        {
            "version": 2,
            "model_state": model.state_dict(),
            "config_snapshot": {
                "model": config.model_dump(mode="json"),
                "training": {"device": "cuda"},
            },
            "identity": identity,
            "step": 9,
            "tokens_processed": 144,
        },
        checkpoint,
    )

    loaded = load_model_checkpoint(checkpoint, device="cpu")

    assert loaded.model.training is False
    assert next(loaded.model.parameters()).device.type == "cpu"
    assert loaded.step == 9
    assert loaded.identity.tokenizer_sha256 == "a" * 64
    for name, value in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], value)

    expected = ArtifactIdentity(
        model_config=config.model_dump(mode="json"),
        tokenizer_sha256="a" * 64,
        corpus_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="checkpoint fields"):
        load_checkpoint(checkpoint, expected)


def test_model_only_loader_maps_weights_only_pickle_failure_to_user_error(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")

    with pytest.raises(CheckpointFormatError, match="invalid checkpoint"):
        load_model_checkpoint(checkpoint, device="cpu")
