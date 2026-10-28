"""Quality and cost measurements for quantized checkpoint exports."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import torch
from torch import nn

from tinyllm.data.loader import TokenDataset
from tinyllm.training.evaluation import EvaluationReport, evaluate_model


@dataclass(frozen=True)
class _ModelMeasurement:
    checkpoint_sha256: str
    artifact_size_bytes: int
    model_state_size_bytes: int
    mean_loss: float
    batch_loss_std: float
    perplexity: float
    latency_ms_per_token: float
    tokens_per_second: float
    average_evaluation_seconds: float
    peak_cuda_memory_bytes: int | None


@dataclass(frozen=True)
class QuantizationComparisonReport:
    schema_version: int
    timestamp_utc: str
    device: str
    evaluation: dict[str, object]
    reference: dict[str, object]
    quantized: dict[str, object]
    quality: dict[str, float]
    methodology: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_quantization_comparison_report(
    report: QuantizationComparisonReport,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("quantization comparison output must use a .json extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"quantization comparison output already exists: {output_path}; pass --overwrite"
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.tmp-",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(report.to_dict(), temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _model_state_size_bytes(model: nn.Module) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_model(
    model: nn.Module,
    dataset: TokenDataset,
    *,
    split: str,
    seed: int,
    batch_size: int,
    batches: int | None,
    tokens: int | None,
    checkpoint_sha256: str,
    corpus_sha256: str,
    artifact_size_bytes: int,
    device: torch.device,
    warmup_runs: int,
    measured_runs: int,
) -> tuple[EvaluationReport, _ModelMeasurement]:
    quality = evaluate_model(
        model,
        dataset,
        split=split,
        seed=seed,
        batch_size=batch_size,
        batches=batches,
        tokens=tokens,
        checkpoint_sha256=checkpoint_sha256,
        corpus_sha256=corpus_sha256,
    )
    baseline_memory = 0
    if device.type == "cuda":
        _synchronize(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup_runs):
        evaluate_model(
            model,
            dataset,
            split=split,
            seed=seed,
            batch_size=batch_size,
            batches=batches,
            tokens=tokens,
            checkpoint_sha256=checkpoint_sha256,
            corpus_sha256=corpus_sha256,
        )

    elapsed = 0.0
    for _ in range(measured_runs):
        _synchronize(device)
        started = perf_counter()
        evaluate_model(
            model,
            dataset,
            split=split,
            seed=seed,
            batch_size=batch_size,
            batches=batches,
            tokens=tokens,
            checkpoint_sha256=checkpoint_sha256,
            corpus_sha256=corpus_sha256,
        )
        _synchronize(device)
        elapsed += perf_counter() - started

    average_seconds = elapsed / measured_runs
    token_count = quality.evaluated_tokens
    peak_memory = None
    if device.type == "cuda":
        peak_memory = max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
    return quality, _ModelMeasurement(
        checkpoint_sha256=checkpoint_sha256,
        artifact_size_bytes=artifact_size_bytes,
        model_state_size_bytes=_model_state_size_bytes(model),
        mean_loss=quality.mean_loss,
        batch_loss_std=quality.batch_loss_std,
        perplexity=quality.perplexity,
        latency_ms_per_token=average_seconds * 1000 / token_count,
        tokens_per_second=token_count / average_seconds if average_seconds else 0.0,
        average_evaluation_seconds=average_seconds,
        peak_cuda_memory_bytes=peak_memory,
    )


def compare_quantized_models(
    reference_model: nn.Module,
    quantized_model: nn.Module,
    dataset: TokenDataset,
    *,
    split: str,
    seed: int,
    batch_size: int,
    batches: int | None = None,
    tokens: int | None = None,
    reference_checkpoint_sha256: str,
    quantized_checkpoint_sha256: str,
    corpus_sha256: str,
    reference_size_bytes: int,
    quantized_size_bytes: int,
    quantized_manifest: Mapping[str, object],
    device: torch.device | str,
    warmup_runs: int = 1,
    measured_runs: int = 3,
) -> QuantizationComparisonReport:
    """Compare model quality and evaluation cost on the same deterministic windows."""
    if type(warmup_runs) is not int or warmup_runs < 0:
        raise ValueError("warmup_runs must be a non-negative integer")
    if type(measured_runs) is not int or measured_runs <= 0:
        raise ValueError("measured_runs must be a positive integer")
    artifact_sizes = (reference_size_bytes, quantized_size_bytes)
    if any(type(size) is not int or size < 0 for size in artifact_sizes):
        raise ValueError("artifact sizes must be non-negative integers")
    resolved_device = torch.device(device)
    reference_report, reference = _measure_model(
        reference_model,
        dataset,
        split=split,
        seed=seed,
        batch_size=batch_size,
        batches=batches,
        tokens=tokens,
        checkpoint_sha256=reference_checkpoint_sha256,
        corpus_sha256=corpus_sha256,
        artifact_size_bytes=reference_size_bytes,
        device=resolved_device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    quantized_report, quantized = _measure_model(
        quantized_model,
        dataset,
        split=split,
        seed=seed,
        batch_size=batch_size,
        batches=batches,
        tokens=tokens,
        checkpoint_sha256=quantized_checkpoint_sha256,
        corpus_sha256=corpus_sha256,
        artifact_size_bytes=quantized_size_bytes,
        device=resolved_device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    if reference_report.evaluated_tokens != quantized_report.evaluated_tokens:
        raise RuntimeError("model evaluations did not use the same number of target tokens")
    return QuantizationComparisonReport(
        schema_version=1,
        timestamp_utc=reference_report.timestamp_utc,
        device=str(resolved_device),
        evaluation={
            "split": split,
            "seed": seed,
            "requested_tokens": tokens,
            "batches": reference_report.batches,
            "batch_size": batch_size,
            "sequence_length": reference_report.sequence_length,
            "evaluated_sequences": reference_report.evaluated_sequences,
            "evaluated_tokens": reference_report.evaluated_tokens,
            "corpus_sha256": corpus_sha256,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
        },
        reference=asdict(reference),
        quantized={**asdict(quantized), "manifest": dict(quantized_manifest)},
        quality={
            "loss_delta_nats_per_token": quantized.mean_loss - reference.mean_loss,
            "perplexity_ratio": quantized.perplexity / reference.perplexity,
            "latency_ratio": quantized.average_evaluation_seconds
            / reference.average_evaluation_seconds
            if reference.average_evaluation_seconds
            else 0.0,
            "artifact_size_ratio": quantized_size_bytes / reference_size_bytes
            if reference_size_bytes
            else 0.0,
            "model_state_size_ratio": quantized.model_state_size_bytes
            / reference.model_state_size_bytes
            if reference.model_state_size_bytes
            else 0.0,
        },
        methodology=(
            "Both models evaluated with evaluate_model on deterministic random windows "
            "from the selected split using the same seed, batch size, and token budget. "
            "Latency is mean full-evaluation wall time per measured run; CUDA peak reports "
            "additional allocated memory above the loaded-model baseline."
        ),
    )

