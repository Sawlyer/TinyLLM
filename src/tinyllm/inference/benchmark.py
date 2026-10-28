"""Comparable cache-on/cache-off autoregressive inference measurements."""

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from tinyllm.config.schema import GenerationConfig
from tinyllm.inference.generate import generate
from tinyllm.model.transformer import TinyLLM


def benchmark_generation_cache(
    model: TinyLLM,
    prompt_ids: Tensor,
    config: GenerationConfig,
    *,
    warmup_runs: int = 2,
    measured_runs: int = 5,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Compare cached and uncached greedy decoding for identical tokens and weights."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError("benchmark prompt must have shape [1, non-empty sequence]")
    if prompt_ids.device != next(model.parameters()).device:
        raise ValueError("benchmark prompt and model must be on the same device")
    if type(warmup_runs) is not int or warmup_runs < 0:
        raise ValueError("warmup_runs must be a non-negative integer")
    if type(measured_runs) is not int or measured_runs <= 0:
        raise ValueError("measured_runs must be a positive integer")
    if prompt_ids.shape[1] + config.max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt and generation must fit within model context")

    benchmark_config = config.model_copy(
        update={
            "temperature": 0.0,
            "top_k": None,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "eos_token_id": None,
            "stop_token_sequences": [],
        }
    )
    device = prompt_ids.device

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run(use_cache: bool) -> tuple[Tensor, dict[str, float | int | None]]:
        first_token_at: float | None = None

        def mark_first_token() -> None:
            nonlocal first_token_at
            synchronize()
            first_token_at = clock()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize()
        started_at = clock()
        output = generate(
            model,
            prompt_ids,
            benchmark_config,
            use_cache=use_cache,
            on_first_token=mark_first_token,
        )
        synchronize()
        finished_at = clock()
        elapsed = max(finished_at - started_at, 1e-12)
        first_latency = (
            None if first_token_at is None else max(first_token_at - started_at, 1e-12)
        )
        generated_tokens = output.shape[1] - prompt_ids.shape[1]
        if generated_tokens != config.max_new_tokens:
            raise RuntimeError("benchmark generation did not produce the requested token count")
        return output, {
            "time_to_first_token_ms": None if first_latency is None else first_latency * 1000,
            "tokens_per_second": generated_tokens / elapsed,
            "decode_tokens_per_second": (
                0.0
                if first_latency is None or generated_tokens <= 1
                else (generated_tokens - 1) / max(elapsed - first_latency, 1e-12)
            ),
            "elapsed_seconds": elapsed,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            ),
        }

    for _ in range(warmup_runs):
        run(True)
        run(False)

    records: dict[str, list[dict[str, float | int | None]]] = {
        "cached": [],
        "uncached": [],
    }
    outputs: dict[str, Tensor] = {}
    for run_index in range(measured_runs):
        modes = (("cached", True), ("uncached", False))
        if run_index % 2:
            modes = tuple(reversed(modes))
        for name, use_cache in modes:
            output, metrics = run(use_cache)
            outputs[name] = output
            records[name].append(metrics)

    greedy_outputs_equal = torch.equal(outputs["cached"], outputs["uncached"])
    cache_dtype = next(model.parameters()).dtype
    cache_bytes = (
        2
        * model.config.n_layers
        * model.config.n_kv_heads
        * (model.config.d_model // model.config.n_heads)
        * model.config.max_seq_len
        * prompt_ids.shape[0]
        * torch.empty((), dtype=cache_dtype).element_size()
    )

    def summarize(values: list[dict[str, float | int | None]], name: str) -> dict[str, Any]:
        numeric_metrics = (
            "time_to_first_token_ms",
            "tokens_per_second",
            "decode_tokens_per_second",
            "elapsed_seconds",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
        )
        summary = {
            metric: (
                None
                if all(value[metric] is None for value in values)
                else median(value[metric] for value in values if value[metric] is not None)
            )
            for metric in numeric_metrics
        }
        summary["kv_cache_capacity_bytes"] = cache_bytes if name == "cached" else 0
        return summary

    return {
        "schema_version": 1,
        "device": str(device),
        "model_dtype": str(cache_dtype),
        "prompt_tokens": int(prompt_ids.shape[1]),
        "generated_tokens": config.max_new_tokens,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "kv_cache_capacity_tokens": model.config.max_seq_len,
        "greedy_outputs_equal": greedy_outputs_equal,
        "modes": {
            name: summarize(records[name], name) for name in ("cached", "uncached")
        },
    }


def write_benchmark_report(
    report: Mapping[str, Any], output_path: Path, *, overwrite: bool = False
) -> None:
    """Atomically write a JSON benchmark report without replacing by default."""
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("inference benchmark output must use a .json extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"inference benchmark output already exists: {output_path}; pass --overwrite"
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
            json.dump(report, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

