import torch

from tinyllm.config.schema import ModelConfig
from tinyllm.data.loader import build_document_masks
from tinyllm.model.transformer import TinyLLM


def tiny_model() -> TinyLLM:
    return TinyLLM(
        ModelConfig(
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
    ).eval()


def test_document_masks_reset_context_after_eos_and_ignore_cross_document_target():
    inputs = torch.tensor([[4, 5, 2, 7, 8]])
    attention_mask, loss_mask = build_document_masks(inputs, eos_token_id=2)

    assert attention_mask.dtype == torch.bool
    assert attention_mask.shape == (1, 5, 5)
    assert attention_mask[0, 3, 3]
    assert not attention_mask[0, 3, 0]
    assert attention_mask[0, 2, 0]
    assert loss_mask.tolist() == [[True, True, False, True, True]]


def test_model_outputs_for_new_document_do_not_depend_on_previous_document():
    torch.manual_seed(7)
    model = tiny_model()
    first = torch.tensor([[3, 4, 2, 6, 7]])
    changed = torch.tensor([[9, 10, 2, 6, 7]])
    first_mask, _ = build_document_masks(first, eos_token_id=2)
    changed_mask, _ = build_document_masks(changed, eos_token_id=2)

    first_logits = model(first, attention_mask=first_mask).logits
    changed_logits = model(changed, attention_mask=changed_mask).logits

    torch.testing.assert_close(first_logits[:, 3:], changed_logits[:, 3:])


def test_model_loss_ignores_target_crossing_eos_boundary():
    torch.manual_seed(11)
    model = tiny_model()
    inputs = torch.tensor([[3, 4, 2, 6, 7]])
    targets = torch.tensor([[4, 2, 6, 7, 8]])
    attention_mask, loss_mask = build_document_masks(inputs, eos_token_id=2)
    changed_targets = targets.clone()
    changed_targets[0, 2] = 11

    loss = model(inputs, targets, attention_mask=attention_mask, loss_mask=loss_mask).loss
    changed_loss = model(
        inputs, changed_targets, attention_mask=attention_mask, loss_mask=loss_mask
    ).loss

    torch.testing.assert_close(loss, changed_loss)
