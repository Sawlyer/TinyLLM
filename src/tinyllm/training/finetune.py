"""Full-parameter supervised fine-tuning for prompt/completion JSONL data."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from tinyllm.config.schema import TrainingConfig
from tinyllm.model.transformer import TinyLLM
from tinyllm.training.optim import build_optimizer, cosine_lr
from tinyllm.training.precision import PrecisionPolicy


class TextTokenizer(Protocol):
    def encode(self, sequence: str, *, add_special_tokens: bool = True) -> Any: ...


@dataclass(frozen=True, slots=True)
class FineTuningExample:
    token_ids: tuple[int, ...]
    prompt_tokens: int


class SupervisedDataset:
    """Prompt/completion examples with loss applied only to completion targets."""

    def __init__(
        self,
        examples: list[FineTuningExample],
        *,
        pad_token_id: int,
        max_seq_len: int,
        seed: int,
    ) -> None:
        if not examples:
            raise ValueError("supervised dataset must contain at least one example")
        self.examples = tuple(examples)
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self) -> int:
        return len(self.examples)

    def sample_positions(self, batch_size: int) -> Tensor:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return torch.randint(len(self.examples), (batch_size,), generator=self.generator)

    def batch(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(positions, Tensor) or positions.ndim != 1:
            raise ValueError("positions must be a one-dimensional tensor")
        indices = positions.detach().to(device="cpu", dtype=torch.long).tolist()
        if any(index < 0 or index >= len(self.examples) for index in indices):
            raise IndexError("supervised example index is out of range")
        sequence_lengths = [len(self.examples[index].token_ids) - 1 for index in indices]
        if not sequence_lengths:
            raise ValueError("batch must contain at least one example")
        batch_length = max(sequence_lengths)
        inputs = torch.full(
            (len(indices), batch_length), self.pad_token_id, dtype=torch.long
        )
        targets = torch.full_like(inputs, self.pad_token_id)
        loss_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for row, (index, length) in enumerate(zip(indices, sequence_lengths, strict=True)):
            example = self.examples[index]
            tokens = torch.tensor(example.token_ids, dtype=torch.long)
            inputs[row, :length] = tokens[:-1]
            targets[row, :length] = tokens[1:]
            loss_mask[row, example.prompt_tokens - 1 : length] = True
        return inputs, targets, loss_mask


def load_supervised_dataset(
    path: Path,
    tokenizer: TextTokenizer,
    *,
    max_seq_len: int,
    pad_token_id: int,
    eos_token_id: int,
    seed: int,
) -> SupervisedDataset:
    """Read strict JSONL rows with string `prompt` and `completion` fields."""
    path = Path(path)
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")
    if type(pad_token_id) is not int or type(eos_token_id) is not int:
        raise ValueError("pad and EOS token IDs must be integers")
    examples: list[FineTuningExample] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} line {line_number} is not valid JSON") from error
            if not isinstance(row, dict) or set(row) != {"prompt", "completion"}:
                raise ValueError(
                    f"{path} line {line_number} must contain only prompt and completion fields"
                )
            prompt = row["prompt"]
            completion = row["completion"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path} line {line_number} has an empty prompt")
            if not isinstance(completion, str) or not completion.strip():
                raise ValueError(f"{path} line {line_number} has an empty completion")
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            completion_ids = tokenizer.encode(completion, add_special_tokens=False).ids
            if not prompt_ids:
                raise ValueError(f"{path} line {line_number} prompt produced no tokens")
            if not completion_ids:
                raise ValueError(f"{path} line {line_number} completion produced no tokens")
            token_ids = (*prompt_ids, *completion_ids, eos_token_id)
            if len(token_ids) - 1 > max_seq_len:
                raise ValueError(
                    f"{path} line {line_number} exceeds model context of {max_seq_len} tokens"
                )
            examples.append(
                FineTuningExample(token_ids=tuple(token_ids), prompt_tokens=len(prompt_ids))
            )
    if not examples:
        raise ValueError(f"{path} contains no prompt/completion examples")
    return SupervisedDataset(
        examples,
        pad_token_id=pad_token_id,
        max_seq_len=max_seq_len,
        seed=seed,
    )


@dataclass(slots=True)
class FineTuningResult:
    model: nn.Module
    steps: int
    training_tokens: int
    scored_tokens: int
    training_loss: float
    validation_loss: float


def run_supervised_finetuning(
    model: TinyLLM,
    train_dataset: SupervisedDataset,
    validation_dataset: SupervisedDataset,
    config: TrainingConfig,
) -> FineTuningResult:
    """Full fine-tune a model and measure final loss on a separate held-out set."""
    if train_dataset.max_seq_len > model.config.max_seq_len:
        raise ValueError("fine-tuning data context exceeds model context")
    if validation_dataset.max_seq_len > model.config.max_seq_len:
        raise ValueError("validation data context exceeds model context")

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    policy = PrecisionPolicy.create(config.precision, device)
    tuned_model = policy.prepare_model(model.to(device))
    optimizer = build_optimizer(
        tuned_model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    training_model = tuned_model
    if config.compile or policy.requires_compile:
        training_model = torch.compile(training_model)

    training_loss_sum = 0.0
    scored_tokens = 0
    training_tokens = 0
    for step in range(config.max_steps):
        training_model.train()
        learning_rate = cosine_lr(
            step,
            config.warmup_steps,
            config.max_steps,
            config.learning_rate,
            config.min_learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        batches: list[tuple[Tensor, Tensor, Tensor]] = []
        for _ in range(config.gradient_accumulation_steps):
            positions = train_dataset.sample_positions(config.micro_batch_size)
            batches.append(train_dataset.batch(positions))
            training_tokens += sum(
                len(train_dataset.examples[int(index)].token_ids) - 1
                for index in positions.tolist()
            )
        step_scored_tokens = sum(int(mask.sum()) for _, _, mask in batches)
        if step_scored_tokens <= 0:
            raise ValueError("fine-tuning batch contains no completion targets")
        for inputs, targets, loss_mask in batches:
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss_mask = loss_mask.to(device)
            with policy.autocast():
                output = training_model(inputs, targets, loss_mask=loss_mask)
            if output.loss is None or not torch.isfinite(output.loss):
                raise ValueError(f"fine-tuning loss is invalid at step {step + 1}")
            token_count = int(loss_mask.sum())
            (output.loss * (token_count / step_scored_tokens)).backward()
            training_loss_sum += float(output.loss.detach()) * token_count
            scored_tokens += token_count
        policy.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(training_model.parameters(), config.max_grad_norm)
        policy.step(optimizer)

    validation_loss = _evaluate_supervised_dataset(
        training_model,
        validation_dataset,
        batch_size=config.micro_batch_size,
        policy=policy,
        device=device,
    )
    average_training_loss = training_loss_sum / max(scored_tokens, 1)
    return FineTuningResult(
        model=tuned_model,
        steps=config.max_steps,
        training_tokens=training_tokens,
        scored_tokens=scored_tokens,
        training_loss=average_training_loss,
        validation_loss=validation_loss,
    )


def _evaluate_supervised_dataset(
    model: nn.Module,
    dataset: SupervisedDataset,
    *,
    batch_size: int,
    policy: PrecisionPolicy,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    try:
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                positions = torch.arange(start, min(start + batch_size, len(dataset)))
                inputs, targets, loss_mask = dataset.batch(positions)
                inputs = inputs.to(device)
                targets = targets.to(device)
                loss_mask = loss_mask.to(device)
                with policy.autocast():
                    output = model(inputs, targets, loss_mask=loss_mask)
                if output.loss is None or not torch.isfinite(output.loss):
                    raise ValueError("fine-tuning validation loss is invalid")
                batch_tokens = int(loss_mask.sum())
                loss_sum += float(output.loss) * batch_tokens
                token_count += batch_tokens
    finally:
        model.train(was_training)
    return loss_sum / max(token_count, 1)


def write_finetuning_manifest(
    path: Path,
    metadata: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically persist fine-tuning provenance beside the inference checkpoint."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"fine-tuning manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(metadata, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
