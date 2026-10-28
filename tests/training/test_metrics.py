import csv
import json
from datetime import datetime

import pytest

from tinyllm.training.metrics import MetricLogger


def test_metric_logger_exports_timestamped_jsonl_and_csv(tmp_path):
    logger = MetricLogger(tmp_path, run_id="run-123")
    logger.log(1, {"loss": 1.25, "tokens_per_second": 64.0})
    logger.close()

    record = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))

    timestamp = datetime.fromisoformat(record["timestamp_utc"].replace("Z", "+00:00"))
    assert timestamp.utcoffset().total_seconds() == 0
    assert record["run_id"] == row["run_id"] == "run-123"
    assert record["session_id"] == row["session_id"]
    assert record["step"] == int(row["step"]) == 1
    assert record["loss"] == float(row["loss"]) == 1.25
    assert record["timestamp_utc"] == row["timestamp_utc"]


def test_metric_logger_expands_csv_columns_without_dropping_rows(tmp_path):
    logger = MetricLogger(tmp_path, run_id="run-123")
    logger.log(1, {"loss": 1.0})
    logger.log(2, {"loss": 0.8, "validation_loss": 0.9})
    logger.close()

    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 2
    assert rows[0]["step"] == "1"
    assert rows[0]["validation_loss"] == ""
    assert float(rows[1]["validation_loss"]) == 0.9


def test_metric_logger_separates_resume_sessions_and_rejects_duplicate_steps(tmp_path):
    first = MetricLogger(tmp_path, run_id="run-123")
    first.log(4, {"loss": 1.0})
    with pytest.raises(ValueError, match="strictly increasing"):
        first.log(4, {"loss": 0.9})
    first.close()

    resumed = MetricLogger(tmp_path, run_id="run-123")
    resumed.log(4, {"loss": 0.9})
    resumed.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert records[0]["step"] == records[1]["step"] == 4
    assert records[0]["session_id"] != records[1]["session_id"]


def test_metric_logger_does_not_duplicate_jsonl_after_csv_error(monkeypatch, tmp_path):
    logger = MetricLogger(tmp_path, run_id="run-123")

    def fail_csv(_event):
        raise OSError("CSV write failed")

    monkeypatch.setattr(logger, "_append_csv", fail_csv)
    with pytest.raises(OSError, match="CSV write failed"):
        logger.log(1, {"loss": 1.0})
    with pytest.raises(ValueError, match="strictly increasing"):
        logger.log(1, {"loss": 1.0})
    logger.close()

    records = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
