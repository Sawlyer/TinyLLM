from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor, nn

from tinyllm.data.loader import TokenDataset


@dataclass(frozen=True)
class EvaluationReport:
    timestamp_utc: str
    split: str
    seed: int
    requested_tokens: int | None
    batches: int
    batch_size: int
    sequence_length: int
    evaluated_sequences: int
    evaluated_tokens: int
    mean_loss: float
    batch_loss_std: float
    perplexity: float
    perplexity_saturated: bool
    checkpoint_sha256: str
    corpus_sha256: str


def evaluate_model(
    model: nn.Module,
    dataset: TokenDataset,
    *,
    split: str,
    seed: int,
    batch_size: int,
    batches: int | None = None,
    tokens: int | None = None,
    checkpoint_sha256: str = "",
    corpus_sha256: str = "",
) -> EvaluationReport:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if getattr(dataset, "split", None) != split:
        raise ValueError("requested split does not match dataset split")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (batches is None) == (tokens is None):
        raise ValueError("select exactly one of batches or tokens")
    if batches is not None and (type(batches) is not int or batches <= 0):
        raise ValueError("batches must be a positive integer")
    if tokens is not None and (type(tokens) is not int or tokens <= 0):
        raise ValueError("tokens must be a positive integer")

    sequence_length = dataset.sequence_length
    if type(sequence_length) is not int or sequence_length <= 0 or len(dataset) <= 0:
        raise ValueError("dataset must contain valid token windows")
    if batches is not None:
        sequence_count = batches * batch_size
    else:
        assert tokens is not None
        sequence_count = (tokens + sequence_length - 1) // sequence_length
    generator = torch.Generator(device="cpu").manual_seed(seed)

    was_training = model.training
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = next(model.buffers(), torch.empty(0)).device
    model.eval()
    batch_losses: list[float] = []
    weighted_loss = 0.0
    evaluated_tokens = 0
    cursor = 0
    try:
        with torch.inference_mode():
            while cursor < sequence_count:
                if tokens is None:
                    row_count = min(batch_size, sequence_count - cursor)
                    token_limit = row_count * sequence_length
                else:
                    remaining = tokens - evaluated_tokens
                    full_rows = min(batch_size, remaining // sequence_length)
                    if full_rows:
                        row_count = full_rows
                        token_limit = row_count * sequence_length
                    else:
                        row_count = 1
                        token_limit = remaining

                positions = torch.randint(
                    low=0,
                    high=len(dataset),
                    size=(row_count,),
                    generator=generator,
                    dtype=torch.long,
                )
                inputs, targets = dataset.batch(positions)
                if token_limit < targets.numel():
                    inputs = inputs.reshape(-1)[:token_limit].reshape(1, token_limit)
                    targets = targets.reshape(-1)[:token_limit].reshape(1, token_limit)
                inputs = inputs.to(model_device)
                targets = targets.to(inputs.device)
                output = model(inputs, targets)
                if output.loss is None or output.loss.ndim != 0:
                    raise ValueError("evaluation must return a scalar next-token loss")
                if not bool(torch.isfinite(output.loss)):
                    raise ValueError("evaluation produced a non-finite loss")
                count = targets.numel()
                batch_loss = float(output.loss)
                batch_losses.append(batch_loss)
                weighted_loss += batch_loss * count
                evaluated_tokens += count
                cursor += row_count
    finally:
        model.train(was_training)

    mean_loss = weighted_loss / evaluated_tokens
    batch_std = statistics.pstdev(batch_losses) if len(batch_losses) > 1 else 0.0
    return EvaluationReport(
        timestamp_utc=_utc_timestamp(),
        split=split,
        seed=seed,
        requested_tokens=tokens,
        batches=len(batch_losses),
        batch_size=batch_size,
        sequence_length=sequence_length,
        evaluated_sequences=sequence_count,
        evaluated_tokens=evaluated_tokens,
        mean_loss=mean_loss,
        batch_loss_std=batch_std,
        perplexity=math.exp(min(mean_loss, 80.0)),
        perplexity_saturated=mean_loss > 80.0,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )


def write_evaluation_report(
    report: EvaluationReport,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in {".json", ".csv"}:
        raise ValueError("evaluation output must use a .json or .csv extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"evaluation output already exists: {output_path}; pass --overwrite")

    fields = asdict(report)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.tmp-",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            if suffix == ".json":
                json.dump(fields, temporary, indent=2, sort_keys=True)
                temporary.write("\n")
            else:
                writer = csv.DictWriter(temporary, fieldnames=list(fields))
                writer.writeheader()
                writer.writerow(fields)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
