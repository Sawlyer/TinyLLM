"""Explicit, preview-first checkpoint listing and retention planning."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tinyllm.errors import CheckpointFormatError
from tinyllm.training.checkpoint import CheckpointInspection, inspect_checkpoint

_STEP_CHECKPOINT = re.compile(r"^step-(?P<step>[0-9]+)\.pt$")


@dataclass(frozen=True, slots=True)
class CheckpointFile:
    """File-system metadata for one direct child checkpoint."""

    path: Path
    size_bytes: int
    modified_time_ns: int
    step: int | None


@dataclass(frozen=True, slots=True)
class CheckpointDecision:
    """Retention decision displayed before any deletion."""

    path: Path
    step: int | None
    size_bytes: int
    resumable: bool | None
    keep_reasons: tuple[str, ...]
    inspection_error: str | None = None

    @property
    def keep(self) -> bool:
        return bool(self.keep_reasons)


def list_checkpoint_files(directory: Path) -> tuple[CheckpointFile, ...]:
    """List direct-child .pt files without deserializing their contents."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {root}")
    files: list[CheckpointFile] = []
    for path in root.iterdir():
        if path.suffix != ".pt" or not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"checkpoint symlinks are not supported: {path}")
        metadata = path.stat()
        match = _STEP_CHECKPOINT.fullmatch(path.name)
        files.append(
            CheckpointFile(
                path=path,
                size_bytes=metadata.st_size,
                modified_time_ns=metadata.st_mtime_ns,
                step=None if match is None else int(match.group("step")),
            )
        )
    return tuple(
        sorted(files, key=lambda item: (item.step is None, item.step or 0, item.path.name))
    )


def plan_checkpoint_retention(
    directory: Path,
    *,
    metrics_file: Path,
    metric: str = "validation_loss",
    mode: Literal["min", "max"] = "min",
    keep_last: int = 2,
    keep_every: int | None = None,
    run_id: str | None = None,
) -> tuple[CheckpointDecision, ...]:
    """Plan deterministic pruning; callers must explicitly apply returned deletions."""
    if type(keep_last) is not int or keep_last < 1:
        raise ValueError("keep_last must be an integer greater than or equal to 1")
    if keep_every is not None and (type(keep_every) is not int or keep_every < 1):
        raise ValueError("keep_every must be an integer greater than or equal to 1")
    if not metric or mode not in {"min", "max"}:
        raise ValueError("metric must be non-empty and mode must be 'min' or 'max'")

    files = list_checkpoint_files(directory)
    step_files = [item for item in files if item.step is not None]
    inspections: dict[Path, CheckpointInspection] = {}
    errors: dict[Path, str] = {}
    mismatched_steps: set[Path] = set()
    for item in step_files:
        try:
            inspection = inspect_checkpoint(item.path)
            inspections[item.path] = inspection
            if inspection.step != item.step:
                mismatched_steps.add(item.path)
                errors[item.path] = (
                    f"filename step {item.step} differs from checkpoint step {inspection.step}"
                )
        except (OSError, RuntimeError, TypeError, ValueError, CheckpointFormatError) as error:
            errors[item.path] = str(error)

    valid = [
        item
        for item in step_files
        if (
            item.path in inspections
            and item.path not in mismatched_steps
            and inspections[item.path].status == "resumable"
        )
    ]
    if not valid:
        raise ValueError("no valid resumable step checkpoint found; nothing was deleted")

    metric_by_step = _read_metrics(metrics_file, metric=metric, run_id=run_id)
    valid.sort(key=lambda item: (item.step or 0, item.path.name))
    selected: dict[Path, list[str]] = {}
    selected[valid[-1].path] = ["latest valid checkpoint"]
    for item in valid[-keep_last:]:
        selected.setdefault(item.path, []).append(f"last {keep_last}")
    if keep_every is not None:
        for item in valid:
            if item.step is not None and item.step % keep_every == 0:
                selected.setdefault(item.path, []).append(f"every {keep_every} steps")

    metric_candidates = [item for item in valid if item.step in metric_by_step]
    if not metric_candidates:
        raise ValueError(
            f"metrics file {metrics_file} has no {metric!r} value for a valid checkpoint step"
        )
    best = min(
        metric_candidates,
        key=lambda item: (
            metric_by_step[item.step],
            item.step or 0,
            item.path.name,
        ),
    ) if mode == "min" else max(
        metric_candidates,
        key=lambda item: (
            metric_by_step[item.step],
            item.step or 0,
            item.path.name,
        ),
    )
    selected.setdefault(best.path, []).append(f"best {metric} ({mode})")

    decisions: list[CheckpointDecision] = []
    for item in files:
        if item.step is None:
            decisions.append(
                CheckpointDecision(
                    item.path,
                    None,
                    item.size_bytes,
                    None,
                    ("named checkpoint preserved",),
                )
            )
            continue
        inspection = inspections.get(item.path)
        reasons = selected.get(item.path, [])
        if item.path in mismatched_steps:
            reasons = ["checkpoint preserved: filename step mismatch"]
        elif inspection is not None and inspection.status != "resumable":
            reasons = ["diagnostic checkpoint preserved"]
        decisions.append(
            CheckpointDecision(
                path=item.path,
                step=item.step,
                size_bytes=item.size_bytes,
                resumable=None if inspection is None else inspection.status == "resumable",
                keep_reasons=tuple(reasons),
                inspection_error=errors.get(item.path),
            )
        )
    if not any(
        decision.keep and decision.resumable is True for decision in decisions
    ):
        raise ValueError("retention plan would remove every valid checkpoint; nothing was deleted")
    return tuple(decisions)


def _read_metrics(
    metrics_file: Path,
    *,
    metric: str,
    run_id: str | None,
) -> dict[int, float]:
    path = Path(metrics_file)
    if not path.is_file():
        raise FileNotFoundError(f"metrics JSONL file does not exist: {path}")
    values: dict[int, float] = {}
    run_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid metrics JSON on line {line_number}: {path}") from error
            if not isinstance(event, dict):
                raise ValueError(f"metrics entry on line {line_number} must be an object")
            event_run_id = event.get("run_id")
            if isinstance(event_run_id, str):
                run_ids.add(event_run_id)
            if run_id is not None and event_run_id != run_id:
                continue
            effective_step = event.get("validation_step", event.get("step"))
            value = event.get(metric)
            if type(effective_step) is not int or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            values[effective_step] = float(value)
    if run_id is None and len(run_ids) > 1:
        raise ValueError("metrics file contains multiple runs; specify --run-id")
    return values

