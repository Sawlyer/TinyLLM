from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    """Write finite scalar metrics to durable JSONL, CSV and TensorBoard streams."""

    _BASE_CSV_FIELDS = ["timestamp_utc", "run_id", "session_id", "step"]

    def __init__(self, run_dir: Path, *, run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_fields = self._read_csv_fields()
        self._csv = self._csv_path.open("a", encoding="utf-8", newline="")
        self._csv_writer = self._create_csv_writer()
        self._writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
        self.run_id = run_id or uuid.uuid4().hex
        self.session_id = uuid.uuid4().hex
        self._last_step: int | None = None
        self._closed = False

    def log(self, step: int, metrics: Mapping[str, int | float]) -> None:
        if self._closed:
            raise RuntimeError("metric logger is closed")
        if type(step) is not int or step < 0:
            raise ValueError("metric step must be a non-negative integer")
        if self._last_step is not None and step <= self._last_step:
            raise ValueError("metric steps must be strictly increasing within a session")
        payload: dict[str, int | float] = {}
        for name, value in metrics.items():
            if name in self._BASE_CSV_FIELDS:
                raise ValueError(f"metric name {name!r} is reserved")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"metric {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"metric {name!r} is non-finite at step {step}")
            payload[name] = value
        event: dict[str, str | int | float] = {
            "timestamp_utc": _utc_timestamp(),
            "run_id": self.run_id,
            "session_id": self.session_id,
            "step": step,
            **payload,
        }
        self._jsonl.write(json.dumps(event, sort_keys=True) + "\n")
        self._last_step = step
        self._jsonl.flush()
        self._append_csv(event)
        for name, value in payload.items():
            self._writer.add_scalar(name, value, step)
        self._writer.flush()

    def _read_csv_fields(self) -> list[str] | None:
        if not self._csv_path.is_file() or self._csv_path.stat().st_size == 0:
            return None
        with self._csv_path.open(encoding="utf-8", newline="") as stream:
            fields = next(csv.reader(stream), None)
        return fields or None

    def _create_csv_writer(self) -> csv.DictWriter:
        return csv.DictWriter(self._csv, fieldnames=self._csv_fields or [])

    def _append_csv(self, event: Mapping[str, str | int | float]) -> None:
        if self._csv_fields is None:
            metric_fields = sorted(set(event) - set(self._BASE_CSV_FIELDS))
            self._csv_fields = [*self._BASE_CSV_FIELDS, *metric_fields]
            self._csv_writer = self._create_csv_writer()
            self._csv_writer.writeheader()
        else:
            missing = [name for name in event if name not in self._csv_fields]
            if missing:
                self._expand_csv_fields([*self._csv_fields, *missing])
        self._csv_writer.writerow(event)
        self._csv.flush()

    def _expand_csv_fields(self, fields: list[str]) -> None:
        self._csv.flush()
        self._csv.close()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                prefix=f".{self._csv_path.name}.tmp-",
                dir=self.run_dir,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                writer = csv.DictWriter(temporary, fieldnames=fields)
                writer.writeheader()
                with self._csv_path.open(encoding="utf-8", newline="") as source:
                    for row in csv.DictReader(source):
                        writer.writerow(row)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self._csv_path)
            temporary_path = None
            self._csv_fields = fields
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            self._csv = self._csv_path.open("a", encoding="utf-8", newline="")
            self._csv_writer = self._create_csv_writer()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._writer.close()
        finally:
            try:
                self._jsonl.close()
            finally:
                self._csv.close()
                self._closed = True

    def __enter__(self) -> MetricLogger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
