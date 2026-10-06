import pytest
import torch

from tinyllm.config.schema import ModelConfig
from tinyllm.model.attention import GroupedQueryAttention


@pytest.fixture
def tiny_attention() -> GroupedQueryAttention:
    config = ModelConfig(
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
    torch.manual_seed(7)
    return GroupedQueryAttention(config)


def test_attention_uses_grouped_key_and_value_projections(
    tiny_attention: GroupedQueryAttention,
) -> None:
    assert tiny_attention.q_proj.out_features == 16
    assert tiny_attention.k_proj.out_features == 8
    assert tiny_attention.v_proj.out_features == 8


def test_attention_preserves_shape_and_empty_cache(
    tiny_attention: GroupedQueryAttention,
) -> None:
    x = torch.randn(2, 5, 16)

    y, cache = tiny_attention(x, torch.arange(5))

    assert y.shape == x.shape
    assert cache is None


def test_attention_cannot_see_future_tokens(tiny_attention: GroupedQueryAttention) -> None:
    x = torch.randn(1, 5, 16)
    changed = x.clone()
    changed[:, 4] += 100

    a, _ = tiny_attention(x, torch.arange(5))
    b, _ = tiny_attention(changed, torch.arange(5))

    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5)


def test_attention_backpropagates_finite_gradients(
    tiny_attention: GroupedQueryAttention,
) -> None:
    x = torch.randn(2, 5, 16, requires_grad=True)

    y, _ = tiny_attention(x, torch.arange(5))
    y.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in tiny_attention.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in tiny_attention.parameters())


def test_attention_rejects_wrong_position_count(
    tiny_attention: GroupedQueryAttention,
) -> None:
    x = torch.randn(1, 5, 16)

    with pytest.raises(ValueError, match="positions"):
        tiny_attention(x, torch.arange(4))
