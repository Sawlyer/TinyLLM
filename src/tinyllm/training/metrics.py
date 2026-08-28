from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    """Write finite scalar metrics to durable JSONL and TensorBoard streams."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self._closed = False

    def log(self, step: int, metrics: Mapping[str, int | float]) -> None:
        if self._closed:
            raise RuntimeError("metric logger is closed")
        if type(step) is not int or step < 0:
            raise ValueError("metric step must be a non-negative integer")
        payload: dict[str, int | float] = {"step": step}
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"metric {name!r} is non-finite at step {step}")
            payload[name] = value
            self._writer.add_scalar(name, value, step)
        self._jsonl.write(json.dumps(payload, sort_keys=True) + "\n")
        self._jsonl.flush()
        self._writer.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        self._jsonl.close()
        self._closed = True

    def __enter__(self) -> MetricLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
