"""Short, opt-in PyTorch profiling for real TinyLLM training batches."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import torch


class _MetricCollector:
    def __init__(self) -> None:
        self.rows: dict[int, dict[str, int | float]] = {}

    def log(self, step: int, metrics: Mapping[str, int | float]) -> None:
        self.rows[step] = dict(metrics)

    def close(self) -> None:
        return None


def run_training_profile(
    trainer: Any,
    *,
    warmup_steps: int,
    measured_steps: int,
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Profile warmup and measured optimizer steps without creating trainer logs."""
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    if type(measured_steps) is not int or measured_steps <= 0:
        raise ValueError("measured_steps must be a positive integer")
    unmeasured_steps = 1
    total_steps = unmeasured_steps + warmup_steps + measured_steps
    profiler_steps = torch.profiler.schedule(
        wait=unmeasured_steps,
        warmup=warmup_steps,
        active=measured_steps,
        repeat=1,
    )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if trainer.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(trainer.device)

    collector = _MetricCollector()
    previous_logger = trainer.metric_logger
    trainer.metric_logger = collector
    started_at = perf_counter()
    try:
        with torch.profiler.profile(
            activities=activities,
            schedule=profiler_steps,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            summary = trainer.train(
                max_steps=total_steps,
                on_step_end=lambda _step: profiler.step(),
            )
        profiling_seconds = max(0.0, perf_counter() - started_at)
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_path))
    finally:
        trainer.metric_logger = previous_logger

    measured_rows = [
        collector.rows[step]
        for step in range(unmeasured_steps + warmup_steps + 1, total_steps + 1)
        if step in collector.rows
    ]
    if len(measured_rows) != measured_steps:
        raise RuntimeError(
            "training profile did not capture metrics for every measured step; "
            "set training.log_interval to 1"
        )

    tokens_per_step = (
        trainer.config.training.micro_batch_size
        * trainer.train_dataset.sequence_length
        * trainer.config.training.gradient_accumulation_steps
    )
    step_reports = []
    for row in measured_rows:
        tokens_per_second = float(row.get("tokens_per_second", 0.0))
        step_seconds = tokens_per_step / tokens_per_second if tokens_per_second > 0 else 0.0
        data_loading_seconds = float(row.get("data_loading_seconds", 0.0))
        step_reports.append(
            {
                "step_seconds": step_seconds,
                "tokens_per_second": tokens_per_second,
                "data_loading_seconds": data_loading_seconds,
                "model_and_optimizer_seconds": max(step_seconds - data_loading_seconds, 0.0),
                "validation_seconds": float(row.get("evaluation_seconds", 0.0)),
                "peak_allocated_bytes": int(row.get("peak_vram_bytes", 0)),
            }
        )

    total_step_seconds = sum(row["step_seconds"] for row in step_reports)
    measured_tokens = tokens_per_step * measured_steps
    memory: dict[str, int | None] = {
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in step_reports),
        "current_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "current_reserved_bytes": None,
    }
    if trainer.device.type == "cuda":
        memory.update(
            {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(trainer.device)),
                "current_allocated_bytes": int(torch.cuda.memory_allocated(trainer.device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(trainer.device)),
                "current_reserved_bytes": int(torch.cuda.memory_reserved(trainer.device)),
            }
        )

    events = profiler.key_averages()
    operators = sorted(
        (
            {
                "name": event.key,
                "calls": int(event.count),
                "self_cpu_time_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                "self_device_time_us": float(
                    getattr(
                        event,
                        "self_device_time_total",
                        getattr(event, "self_cuda_time_total", 0.0),
                    )
                ),
            }
            for event in events
        ),
        key=lambda event: event["self_cpu_time_us"],
        reverse=True,
    )
    phase_events = [event for event in operators if event["name"].startswith("tinyllm/")]
    return {
        "unmeasured_steps": unmeasured_steps,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "measurement": {
            "tokens_per_step": tokens_per_step,
            "measured_tokens": measured_tokens,
            "total_step_seconds": total_step_seconds,
            "mean_step_seconds": total_step_seconds / measured_steps,
            "mean_tokens_per_second": measured_tokens / total_step_seconds
            if total_step_seconds > 0
            else 0.0,
            "mean_data_loading_seconds": sum(
                row["data_loading_seconds"] for row in step_reports
            )
            / measured_steps,
            "mean_model_and_optimizer_seconds": sum(
                row["model_and_optimizer_seconds"] for row in step_reports
            )
            / measured_steps,
            "validation_seconds": sum(row["validation_seconds"] for row in step_reports),
            "profiled_wall_seconds": profiling_seconds,
            "trainer_elapsed_seconds": float(summary.elapsed_seconds),
            "profiler_overhead_seconds": max(
                profiling_seconds - float(summary.elapsed_seconds), 0.0
            ),
            "steps": step_reports,
        },
        "memory": memory,
        "phase_events": phase_events,
        "top_operators": operators[:20],
        "trace_file": str(trace_path) if trace_path is not None else None,
    }

