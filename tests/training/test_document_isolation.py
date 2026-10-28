from types import SimpleNamespace

import torch

from tinyllm.training.trainer import Trainer


class CapturingModel:
    def __init__(self) -> None:
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return SimpleNamespace(loss=torch.tensor(0.5))


def test_trainer_passes_document_masks_only_when_isolation_enabled():
    trainer = object.__new__(Trainer)
    trainer.isolate_documents = True
    trainer.eos_token_id = 2
    trainer._training_model = CapturingModel()
    inputs = torch.tensor([[4, 2, 7]])
    targets = torch.tensor([[2, 7, 8]])

    trainer._forward_training_batch(inputs, targets)

    assert trainer._training_model.args == (inputs, targets)
    attention_mask = trainer._training_model.kwargs["attention_mask"]
    loss_mask = trainer._training_model.kwargs["loss_mask"]
    assert not attention_mask[0, 2, 0]
    assert loss_mask.tolist() == [[True, False, True]]


def test_trainer_preserves_legacy_model_call_when_isolation_is_disabled():
    trainer = object.__new__(Trainer)
    trainer.isolate_documents = False
    trainer._training_model = CapturingModel()
    inputs = torch.tensor([[4, 2, 7]])
    targets = torch.tensor([[2, 7, 8]])

    trainer._forward_training_batch(inputs, targets)

    assert trainer._training_model.args == (inputs, targets)
    assert trainer._training_model.kwargs == {}
