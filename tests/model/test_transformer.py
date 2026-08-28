import math

import pytest
import torch
import torch.nn.functional as F

from tinyllm.config.schema import ModelConfig
from tinyllm.model.cache import KVCache
from tinyllm.model.transformer import ModelOutput, TinyLLM, count_parameters


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        max_seq_len=16,
        d_model=16,
        n_layers=2,
        n_heads=8,
        n_kv_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        rope_theta=10_000.0,
        rms_norm_eps=1e-5,
    )


@pytest.fixture
def tiny_model(tiny_config: ModelConfig) -> TinyLLM:
    torch.manual_seed(7)
    return TinyLLM(tiny_config)


def test_language_model_returns_loss_and_logits(tiny_model: TinyLLM) -> None:
    ids = torch.randint(0, 64, (2, 8))

    out = tiny_model(ids, targets=ids)

    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (2, 8, 64)
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert torch.allclose(
        out.loss,
        F.cross_entropy(out.logits.reshape(-1, 64), ids.reshape(-1)),
    )
    assert out.caches is None


def test_language_model_omits_loss_without_targets(tiny_model: TinyLLM) -> None:
    out = tiny_model(torch.randint(0, 64, (2, 8)))

    assert out.loss is None


def test_language_model_rejects_empty_batch_before_loss(tiny_model: TinyLLM) -> None:
    ids = torch.empty((0, 8), dtype=torch.long)

    with pytest.raises(ValueError, match="input batch must contain at least one sequence"):
        tiny_model(ids, targets=ids)


def test_embedding_and_output_weights_are_tied(tiny_model: TinyLLM) -> None:
    assert tiny_model.lm_head.weight is tiny_model.token_embedding.weight


def test_language_model_is_causal(tiny_model: TinyLLM) -> None:
    ids = torch.randint(0, 64, (1, 8))
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 64

    original = tiny_model(ids).logits
    modified = tiny_model(changed).logits

    assert torch.allclose(original[:, :-1], modified[:, :-1], atol=1e-5)


def test_residual_projections_use_depth_scaled_initialization(tiny_model: TinyLLM) -> None:
    residual_weights = torch.cat(
        [
            parameter.detach().flatten()
            for name, parameter in tiny_model.named_parameters()
            if name.endswith(("attention.out_proj.weight", "feed_forward.down_proj.weight"))
        ]
    )
    ordinary_weights = torch.cat(
        [
            parameter.detach().flatten()
            for name, parameter in tiny_model.named_parameters()
            if name.endswith("attention.q_proj.weight")
        ]
    )

    expected_residual_std = 0.02 / math.sqrt(2 * tiny_model.config.n_layers)
    assert residual_weights.std().item() == pytest.approx(expected_residual_std, rel=0.08)
    assert ordinary_weights.std().item() == pytest.approx(0.02, rel=0.15)


def test_parameter_count_deduplicates_tied_weights(tiny_model: TinyLLM) -> None:
    total, trainable = count_parameters(tiny_model)

    assert total == 5_712
    assert trainable == total


def test_language_model_backpropagates_finite_gradients(tiny_model: TinyLLM) -> None:
    ids = torch.randint(0, 64, (2, 8))
    out = tiny_model(ids, targets=ids)
    assert out.loss is not None

    out.loss.backward()

    assert all(parameter.grad is not None for parameter in tiny_model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in tiny_model.parameters()
        if parameter.grad is not None
    )


def test_language_model_returns_one_updated_cache_per_layer(tiny_model: TinyLLM) -> None:
    ids = torch.randint(0, 64, (1, 4))

    output = tiny_model(ids, caches=[KVCache(), KVCache()])

    assert output.caches is not None
    assert [cache.length for cache in output.caches] == [4, 4]


def test_language_model_rejects_wrong_cache_count(tiny_model: TinyLLM) -> None:
    ids = torch.randint(0, 64, (1, 4))

    with pytest.raises(ValueError, match="one cache per layer"):
        tiny_model(ids, caches=[None])
