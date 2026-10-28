from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tinyllm.training import checkpoint_manager
from tinyllm.training.checkpoint import CheckpointInspection


def _inspection(path: Path, step: int) -> CheckpointInspection:
    return CheckpointInspection(
        path=path,
        size_bytes=path.stat().st_size,
        step=step,
        tokens_processed=step * 16,
        status="resumable",
        reason=None,
        model_config={"layers": 1},
        tokenizer_sha256="tokenizer",
        corpus_sha256="corpus",
    )


def _write_metrics(path: Path, values: dict[int, float]) -> None:
    rows = [
        {
            "run_id": "run-a",
            "step": step,
            "validation_step": step,
            "validation_loss": value,
        }
        for step, value in values.items()
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_list_checkpoint_files_does_not_deserialize_and_ignores_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step_checkpoint = tmp_path / "step-00000010.pt"
    step_checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "latest.pt").write_bytes(b"latest")
    (tmp_path / ".latest.pt.random.tmp").write_bytes(b"temporary")
    (tmp_path / "not-a-checkpoint.txt").write_text("text", encoding="utf-8")
    monkeypatch.setattr(
        checkpoint_manager,
        "inspect_checkpoint",
        lambda _: pytest.fail("listing must not deserialize checkpoint contents"),
    )

    files = checkpoint_manager.list_checkpoint_files(tmp_path)

    assert [(item.path.name, item.step) for item in files] == [
        ("step-00000010.pt", 10),
        ("latest.pt", None),
    ]


def test_retention_keeps_latest_best_spaced_and_named_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints = {step: tmp_path / f"step-{step:08d}.pt" for step in (10, 20, 30, 40)}
    for path in checkpoints.values():
        path.write_bytes(b"x")
    named_checkpoint = tmp_path / "latest.pt"
    named_checkpoint.write_bytes(b"x")
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics, {10: 1.2, 20: 0.8, 30: 1.0, 40: 1.1})
    monkeypatch.setattr(
        checkpoint_manager,
        "inspect_checkpoint",
        lambda path: _inspection(path, int(path.stem.removeprefix("step-"))),
    )

    plan = checkpoint_manager.plan_checkpoint_retention(
        tmp_path,
        metrics_file=metrics,
        keep_last=1,
        keep_every=30,
        run_id="run-a",
    )
    decisions = {item.path.name: item for item in plan}

    assert decisions["step-00000010.pt"].keep is False
    assert "best validation_loss (min)" in decisions["step-00000020.pt"].keep_reasons
    assert "every 30 steps" in decisions["step-00000030.pt"].keep_reasons
    assert "latest valid checkpoint" in decisions["step-00000040.pt"].keep_reasons
    assert decisions["latest.pt"].keep_reasons == ("named checkpoint preserved",)
    assert all(path.exists() for path in checkpoints.values())


def test_retention_never_deletes_when_no_resumable_checkpoint_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "step-00000010.pt"
    checkpoint.write_bytes(b"invalid")
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics, {10: 1.0})
    monkeypatch.setattr(
        checkpoint_manager,
        "inspect_checkpoint",
        lambda _: (_ for _ in ()).throw(ValueError("invalid checkpoint")),
    )

    with pytest.raises(ValueError, match="no valid resumable step checkpoint"):
        checkpoint_manager.plan_checkpoint_retention(tmp_path, metrics_file=metrics)

    assert checkpoint.exists()


def test_retention_requires_run_id_for_metrics_from_multiple_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "step-00000010.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        '{"run_id":"run-a","step":10,"validation_step":10,"validation_loss":1.0}\n'
        '{"run_id":"run-b","step":10,"validation_step":10,"validation_loss":0.9}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        checkpoint_manager,
        "inspect_checkpoint",
        lambda path: _inspection(path, 10),
    )

    with pytest.raises(ValueError, match="multiple runs; specify --run-id"):
        checkpoint_manager.plan_checkpoint_retention(tmp_path, metrics_file=metrics)


def test_retention_preserves_checkpoint_with_filename_step_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mismatched = tmp_path / "step-00000010.pt"
    valid = tmp_path / "step-00000020.pt"
    mismatched.write_bytes(b"checkpoint")
    valid.write_bytes(b"checkpoint")
    metrics = tmp_path / "metrics.jsonl"
    _write_metrics(metrics, {20: 0.8})

    def inspect(path: Path) -> CheckpointInspection:
        return _inspection(path, 11 if path == mismatched else 20)

    monkeypatch.setattr(checkpoint_manager, "inspect_checkpoint", inspect)
    plan = checkpoint_manager.plan_checkpoint_retention(tmp_path, metrics_file=metrics)
    decisions = {item.path.name: item for item in plan}

    assert decisions[mismatched.name].keep_reasons == (
        "checkpoint preserved: filename step mismatch",
    )
    assert decisions[valid.name].keep is True

