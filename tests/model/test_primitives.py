import pytest
import torch

from tinyllm.config.schema import ModelConfig
from tinyllm.model.mlp import SwiGLU
from tinyllm.model.norm import RMSNorm
from tinyllm.model.rope import RotaryEmbedding


def _model_config() -> ModelConfig:
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


def test_rms_norm_has_unit_rms() -> None:
    y = RMSNorm(4)(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    assert torch.allclose(y.square().mean(-1), torch.ones(1), atol=1e-5)


def test_rms_norm_preserves_shape_dtype_and_gradients() -> None:
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    norm = RMSNorm(4).double()

    y = norm(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()


def test_rotary_embedding_preserves_shape_norm_and_position_zero() -> None:
    rope = RotaryEmbedding(dim=4, max_seq_len=8, theta=10_000.0)
    x = torch.randn(2, 3, 5, 4, requires_grad=True)

    y = rope(x, torch.arange(5))

    assert y.shape == x.shape
    assert torch.allclose(y[..., 0, :], x[..., 0, :])
    assert torch.allclose(y.square().sum(-1), x.square().sum(-1), atol=1e-5)
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_rotary_embedding_matches_known_rotation() -> None:
    rope = RotaryEmbedding(dim=2, max_seq_len=4)
    x = torch.tensor([[[[1.0, 2.0]]]])

    y = rope(x, torch.tensor([1]))

    expected = torch.tensor([[[[-1.1426396, 1.9220756]]]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_rotary_embedding_supports_per_batch_positions() -> None:
    rope = RotaryEmbedding(dim=4, max_seq_len=8)
    x = torch.randn(2, 3, 2, 4)
    positions = torch.tensor([[0, 1], [2, 3]])

    y = rope(x, positions)

    assert y.shape == x.shape
    assert torch.allclose(y[0, :, 0], x[0, :, 0])


def test_rotary_embedding_rejects_non_four_dimensional_input() -> None:
    rope = RotaryEmbedding(dim=4, max_seq_len=8)
    x = torch.randn(2, 5, 4)

    with pytest.raises(ValueError, match="four dimensions"):
        rope(x, torch.arange(5))


def test_rotary_embedding_rejects_floating_point_positions() -> None:
    rope = RotaryEmbedding(dim=4, max_seq_len=8)
    x = torch.randn(1, 2, 3, 4)

    with pytest.raises(TypeError, match="integer dtype"):
        rope(x, torch.tensor([0.5, 1.5, 2.5]))


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_rotary_embedding_compiles_as_full_graph() -> None:
    rope = RotaryEmbedding(dim=4, max_seq_len=8)
    x = torch.randn(2, 3, 5, 4)
    positions = torch.arange(5)
    compiled = torch.compile(rope, backend="eager", fullgraph=True)

    actual = compiled(x, positions)
    expected = rope(x, positions)

    assert torch.allclose(actual, expected)


def test_swiglu_preserves_model_width_and_backpropagates() -> None:
    mlp = SwiGLU(_model_config())
    x = torch.randn(2, 5, 16, requires_grad=True)

    y = mlp(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(parameter.grad is not None for parameter in mlp.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in mlp.parameters())


def test_swiglu_matches_known_gate_values() -> None:
    config = _model_config().model_copy(update={"d_model": 2, "mlp_ratio": 1.0})
    mlp = SwiGLU(config)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.eye(2))
        mlp.up_proj.weight.copy_(torch.eye(2))
        mlp.down_proj.weight.copy_(torch.eye(2))

    y = mlp(torch.tensor([[[-1.0, 2.0]]]))

    expected = torch.tensor([[[0.2689414, 3.5231884]]])
    assert torch.allclose(y, expected, atol=1e-6)
