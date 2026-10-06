import pytest
import torch

from tinyllm.config.schema import GenerationConfig
from tinyllm.inference.sampling import sample_next


def test_zero_temperature_selects_greedy_token() -> None:
    logits = torch.tensor([[0.1, 2.0, 0.3]])

    sampled = sample_next(
        logits,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        generator=torch.Generator().manual_seed(3),
    )

    assert torch.equal(sampled, torch.tensor([[1]]))


def test_generation_config_allows_zero_temperature_for_greedy_decoding() -> None:
    config = GenerationConfig(
        max_new_tokens=2,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        seed=3,
    )

    assert config.temperature == 0.0


def test_sampling_is_seeded() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.3]]).expand(64, -1)

    first = sample_next(
        logits,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        generator=torch.Generator().manual_seed(9),
    )
    second = sample_next(
        logits,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        generator=torch.Generator().manual_seed(9),
    )

    assert torch.equal(first, second)


def test_temperature_changes_sample_concentration() -> None:
    logits = torch.tensor([[0.0, 1.0]]).expand(1024, -1)

    cold = sample_next(
        logits,
        temperature=0.1,
        top_k=None,
        top_p=1.0,
        generator=torch.Generator().manual_seed(12),
    )
    hot = sample_next(
        logits,
        temperature=10.0,
        top_k=None,
        top_p=1.0,
        generator=torch.Generator().manual_seed(12),
    )

    assert cold.sum().item() > hot.sum().item() + 300


def test_top_k_excludes_tokens_below_cutoff() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(256, -1)

    sampled = sample_next(
        logits,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        generator=torch.Generator().manual_seed(4),
    )

    assert set(sampled.flatten().tolist()) == {2, 3}


def test_top_k_keeps_exactly_k_indices_when_cutoff_logits_are_tied() -> None:
    logits = torch.tensor([[3.0, 2.0, 2.0, 2.0]]).expand(512, -1)

    sampled = sample_next(
        logits,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        generator=torch.Generator().manual_seed(8),
    )

    sampled_ids = set(sampled.flatten().tolist())
    assert 0 in sampled_ids
    assert len(sampled_ids) == 2


def test_top_p_excludes_tokens_outside_nucleus() -> None:
    logits = torch.zeros(256, 4)

    sampled = sample_next(
        logits,
        temperature=1.0,
        top_k=None,
        top_p=0.5,
        generator=torch.Generator().manual_seed(5),
    )

    assert set(sampled.flatten().tolist()) == {0, 1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": -1.0, "top_k": None, "top_p": 1.0}, "temperature"),
        ({"temperature": 1.0, "top_k": 0, "top_p": 1.0}, "top_k"),
        ({"temperature": 1.0, "top_k": None, "top_p": 0.0}, "top_p"),
        ({"temperature": 1.0, "top_k": None, "top_p": 1.1}, "top_p"),
    ],
)
def test_sampling_rejects_invalid_controls(
    kwargs: dict[str, float | int | None],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_next(
            torch.ones(1, 3),
            generator=torch.Generator().manual_seed(0),
            **kwargs,
        )


def test_sampling_rejects_nan_logits_with_actionable_error() -> None:
    with pytest.raises(ValueError, match="NaN"):
        sample_next(
            torch.tensor([[0.0, float("nan")]]),
            temperature=1.0,
            top_k=None,
            top_p=1.0,
            generator=torch.Generator().manual_seed(0),
        )


def test_sampling_rejects_rows_without_finite_logits() -> None:
    with pytest.raises(ValueError, match="finite logit"):
        sample_next(
            torch.tensor([[float("-inf"), float("-inf")]]),
            temperature=1.0,
            top_k=None,
            top_p=1.0,
            generator=torch.Generator().manual_seed(0),
        )
