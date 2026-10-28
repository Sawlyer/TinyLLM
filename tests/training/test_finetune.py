import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tinyllm.cli import _build_parser
from tinyllm.config.schema import ModelConfig, TrainingConfig
from tinyllm.training.finetune import (
    load_supervised_dataset,
    run_supervised_finetuning,
    write_finetuning_manifest,
)
from tinyllm.model.transformer import TinyLLM


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        token_ids = {"user": 1, "assistant": 2, "hello": 3, "world": 4}
        return SimpleNamespace(ids=[token_ids[token] for token in text.split()])


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_supervised_dataset_masks_prompt_and_keeps_eos_in_loss(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"prompt": "user hello", "completion": "assistant world"}])

    dataset = load_supervised_dataset(
        path, FakeTokenizer(), max_seq_len=8, pad_token_id=0, eos_token_id=5, seed=17
    )
    inputs, targets, loss_mask = dataset.batch(torch.tensor([0]))

    assert inputs.tolist() == [[1, 3, 2, 4]]
    assert targets.tolist() == [[3, 2, 4, 5]]
    assert loss_mask.tolist() == [[False, True, True, True]]


def test_supervised_dataset_rejects_examples_beyond_context(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(path, [{"prompt": "user hello", "completion": "assistant world"}])

    with pytest.raises(ValueError, match="line 1.*context"):
        load_supervised_dataset(
            path, FakeTokenizer(), max_seq_len=3, pad_token_id=0, eos_token_id=5, seed=17
        )


def test_supervised_dataset_right_pads_without_scoring_padding(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_jsonl(
        path,
        [
            {"prompt": "user", "completion": "hello"},
            {"prompt": "user hello", "completion": "assistant world"},
        ],
    )
    dataset = load_supervised_dataset(
        path, FakeTokenizer(), max_seq_len=8, pad_token_id=0, eos_token_id=5, seed=17
    )

    inputs, targets, loss_mask = dataset.batch(torch.tensor([0, 1]))

    assert inputs.tolist() == [[1, 3, 0, 0], [1, 3, 2, 4]]
    assert targets.tolist() == [[3, 5, 0, 0], [3, 2, 4, 5]]
    assert loss_mask.tolist() == [[False, True, False, False], [False, True, True, True]]


def test_full_finetuning_updates_weights_and_evaluates_separate_split(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(train_path, [{"prompt": "user hello", "completion": "assistant world"}])
    write_jsonl(validation_path, [{"prompt": "user", "completion": "assistant hello"}])
    model_config = ModelConfig(
        vocab_size=8,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        rope_theta=10_000.0,
        rms_norm_eps=1e-5,
    )
    train = load_supervised_dataset(
        train_path, FakeTokenizer(), max_seq_len=8, pad_token_id=0, eos_token_id=5, seed=17
    )
    validation = load_supervised_dataset(
        validation_path, FakeTokenizer(), max_seq_len=8, pad_token_id=0, eos_token_id=5, seed=29
    )
    model = TinyLLM(model_config)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    config = TrainingConfig(
        seed=17,
        device="cpu",
        precision="fp32",
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=2,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        compile=False,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    result = run_supervised_finetuning(model, train, validation, config)

    assert result.steps == 2
    assert result.training_tokens > 0
    assert result.scored_tokens > 0
    assert result.validation_loss > 0
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_finetuning_manifest_is_json_and_refuses_implicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "model.finetune.json"
    write_finetuning_manifest(path, {"base_checkpoint_sha256": "abc"})

    assert json.loads(path.read_text(encoding="utf-8"))["base_checkpoint_sha256"] == "abc"
    with pytest.raises(FileExistsError, match="already exists"):
        write_finetuning_manifest(path, {"base_checkpoint_sha256": "changed"})


def test_finetune_cli_exposes_separate_data_and_checkpoint_paths() -> None:
    args = _build_parser().parse_args(
        [
            "finetune",
            "--checkpoint",
            "base.pt",
            "--train-data",
            "train.jsonl",
            "--validation-data",
            "validation.jsonl",
            "--output",
            "adapted.pt",
            "--max-steps",
            "20",
        ]
    )

    assert args.handler.__name__ == "_finetune_command"
    assert args.max_steps == 20
    assert args.train_data == Path("train.jsonl")
