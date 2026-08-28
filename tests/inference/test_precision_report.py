from contextlib import nullcontext

import pytest
import torch
from torch import nn

from tinyllm.inference.precision_report import benchmark_inference


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        logits = self.projection(torch.nn.functional.one_hot(inputs, 4).float())
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return type("Output", (), {"logits": logits, "loss": loss})()


def test_inference_benchmark_reports_quality_speed_and_reference_error() -> None:
    model = ToyModel()
    batches = [
        (torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]])),
        (torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 0]])),
    ]
    ticks = iter((1.0, 3.0, 4.0, 6.0))
    reference = [model(inputs, targets).logits.detach() for inputs, targets in batches]

    result = benchmark_inference(
        model,
        batches,
        mode="fp32",
        device=torch.device("cpu"),
        autocast_context=nullcontext,
        reference_logits=reference,
        repetitions=2,
        warmup_repetitions=0,
        timer=lambda: next(ticks),
    )

    assert result.mode == "fp32"
    assert result.mean_loss > 0
    assert result.perplexity == pytest.approx(torch.exp(torch.tensor(result.mean_loss)).item())
    assert result.tokens_per_second == (3.0, 3.0)
    assert result.mean_absolute_logit_error == pytest.approx(0.0)
    assert result.max_absolute_logit_error == pytest.approx(0.0)
