import pytest
from torch import nn

from tinyllm.model.norm import RMSNorm
from tinyllm.training.optim import build_optimizer, cosine_lr


def test_cosine_schedule_hits_boundaries() -> None:
    assert cosine_lr(0, 10, 100, 1e-3, 1e-4) == 0.0
    assert cosine_lr(10, 10, 100, 1e-3, 1e-4) == pytest.approx(1e-3)
    assert cosine_lr(100, 10, 100, 1e-3, 1e-4) == pytest.approx(1e-4)


def test_cosine_schedule_without_warmup_starts_at_maximum_and_clamps() -> None:
    assert cosine_lr(0, 0, 10, 1.0, 0.1) == pytest.approx(1.0)
    assert cosine_lr(20, 0, 10, 1.0, 0.1) == pytest.approx(0.1)


def test_cosine_schedule_rejects_warmup_without_decay_interval() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        cosine_lr(9, 10, 10, 1.0, 0.1)


def test_adamw_excludes_biases_and_normalization_weights_from_decay() -> None:
    model = nn.Sequential(nn.Linear(4, 4), RMSNorm(4))

    optimizer = build_optimizer(model, learning_rate=1e-3, weight_decay=0.2)

    decay_by_parameter = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert decay_by_parameter[id(model[0].weight)] == pytest.approx(0.2)
    assert decay_by_parameter[id(model[0].bias)] == 0.0
    assert decay_by_parameter[id(model[1].weight)] == 0.0
    assert len(decay_by_parameter) == len(list(model.parameters()))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"step": -1}, "step"),
        ({"warmup_steps": 11}, "warmup_steps"),
        ({"min_lr": 2.0}, "min_lr"),
    ],
)
def test_cosine_schedule_rejects_invalid_inputs(kwargs: dict[str, float], message: str) -> None:
    arguments = {
        "step": 1,
        "warmup_steps": 2,
        "total_steps": 10,
        "max_lr": 1.0,
        "min_lr": 0.1,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        cosine_lr(**arguments)
