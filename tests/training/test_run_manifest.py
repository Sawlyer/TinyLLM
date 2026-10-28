import argparse
import json
import signal
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from tinyllm.config.load import load_config
from tinyllm.training import run_manifest


def _config():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "tinyllm.yaml"
    return load_config(config_path, [])


def test_run_manifest_has_stable_schema_and_portable_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = _config()
    config = config.model_copy(
        update={
            "data": config.data.model_copy(
                update={"dataset_name": "C:/Users/Alice/private/corpus.jsonl"}
            )
        }
    )
    args = argparse.Namespace(
        command="train",
        max_steps=4,
        checkpoint=Path("C:/Users/Alice/private/checkpoint.pt"),
        resume=None,
        overwrite=False,
        dry_run=False,
    )
    monkeypatch.setattr(run_manifest, "_git_revision", lambda: "abc1234")
    monkeypatch.setattr(
        run_manifest,
        "_runtime_metadata",
        lambda: {"python": "3.12.1", "torch": "2.5.0"},
    )
    monkeypatch.setattr(
        run_manifest,
        "_system_metadata",
        lambda: {"system": "Windows", "release": "11", "machine": "AMD64"},
    )
    monkeypatch.setattr(run_manifest, "_accelerator_name", lambda: "RTX 5080")

    payload = run_manifest.build_run_manifest(
        config,
        args,
        status="running",
        started_at="2026-10-28T12:00:00Z",
        finished_at=None,
    )

    assert set(payload) == {
        "schema_version",
        "run_id",
        "status",
        "started_at",
        "finished_at",
        "source_revision",
        "runtime",
        "system",
        "accelerator",
        "config",
        "arguments",
        "seeds",
        "artifacts",
    }
    encoded = json.dumps(payload)
    assert "C:/Users/Alice" not in encoded
    assert "private" not in encoded
    assert payload["config"]["data"]["dataset_name"] == "<external>/corpus.jsonl"
    assert payload["arguments"]["max_steps"] == 4
    assert payload["source_revision"] == "abc1234"


def test_run_manifest_omits_unavailable_source_and_system_metadata(monkeypatch):
    config = _config()
    args = argparse.Namespace(command="train", max_steps=None)
    monkeypatch.setattr(run_manifest, "_git_revision", lambda: None)
    monkeypatch.setattr(run_manifest, "_runtime_metadata", lambda: {})
    monkeypatch.setattr(run_manifest, "_system_metadata", lambda: {})
    monkeypatch.setattr(run_manifest, "_accelerator_name", lambda: None)

    payload = run_manifest.build_run_manifest(
        config,
        args,
        status="completed",
        started_at="2026-10-28T12:00:00Z",
        finished_at="2026-10-28T12:01:00Z",
    )

    assert "source_revision" not in payload
    assert "runtime" not in payload
    assert "system" not in payload
    assert "accelerator" not in payload


def test_runtime_metadata_records_cuda_runtime_when_available(monkeypatch):
    monkeypatch.setattr(
        run_manifest.importlib.metadata,
        "version",
        lambda distribution: f"{distribution}-test",
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(version=SimpleNamespace(cuda="12.4")),
    )

    assert run_manifest._runtime_metadata()["cuda"] == "12.4"


def test_run_manifest_records_corpus_manifest_and_declared_data_checksums(tmp_path):
    config = _config()
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tokenizer_sha256": "a" * 64,
                "artifacts": {
                    "train": {"sha256": "b" * 64},
                    "validation": {"sha256": "c" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    config = config.model_copy(
        update={"data": config.data.model_copy(update={"output_dir": output_dir})}
    )

    digests = run_manifest._artifact_digests(config, argparse.Namespace())

    assert digests["data.corpus_manifest"] == run_manifest._file_sha256(
        output_dir / "manifest.json"
    )
    assert digests["data.tokenizer"] == "a" * 64
    assert digests["data.train_tokens"] == "b" * 64
    assert digests["data.validation_tokens"] == "c" * 64


def test_run_manifest_preserves_unreadable_prior_manifest(tmp_path):
    config = _config()
    run_dir = tmp_path / "runs"
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"run_dir": run_dir})}
    )
    run_dir.mkdir()
    (run_dir / "run.json").write_text("{invalid", encoding="utf-8")

    run_manifest.start_run_manifest(
        config,
        argparse.Namespace(command="train", resume=None),
        started_at="2026-10-28T12:00:00Z",
    )

    archives = list(run_dir.glob("run-invalid-*.json"))
    assert len(archives) == 1
    assert archives[0].read_text(encoding="utf-8") == "{invalid"


def test_archive_rejects_path_components_in_run_id(tmp_path):
    manifest_path = tmp_path / "run.json"
    manifest_path.write_text('{"run_id":"../../outside"}', encoding="utf-8")

    run_manifest._archive_current_manifest(
        manifest_path, {"run_id": "../../outside"}
    )

    assert not (tmp_path.parent / "outside.json").exists()
    archives = list(tmp_path.glob("run-*.json"))
    assert len(archives) == 1


def test_sigterm_handler_maps_termination_to_interrupted_run(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from tinyllm import cli

    config = _config()
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"run_dir": tmp_path / "runs"})}
    )
    registered = {}
    original_signal = signal.signal

    def capture_signal(signum, handler):
        previous = original_signal(signum, handler)
        if signum == signal.SIGTERM:
            registered[signum] = handler
        return previous

    monkeypatch.setattr(cli.signal, "signal", capture_signal)

    class TerminatingTrainer(_FakeTrainer):
        def train(self, *, max_steps: int | None, overwrite_checkpoints: bool):
            registered[signal.SIGTERM](signal.SIGTERM, None)

    trainer = TerminatingTrainer(tmp_path / "latest.pt")
    _install_cli_fakes(monkeypatch, cli, config, trainer)
    with pytest.raises(KeyboardInterrupt):
        cli._train_command(_train_args(tmp_path / "latest.pt"))

    manifest = json.loads((tmp_path / "runs" / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"


def test_write_run_manifest_replaces_atomically_and_preserves_old_file_on_failure(
    monkeypatch, tmp_path
):
    path = tmp_path / "run.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(run_manifest.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        run_manifest.write_run_manifest(path, {"status": "completed"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob("*.tmp-*")) == []


class _FakeTrainer:
    def __init__(self, checkpoint: Path, failure: BaseException | None = None):
        self.checkpoint = checkpoint
        self.failure = failure

    def load_checkpoint(self, _path: Path) -> None:
        return None

    def train(self, *, max_steps: int | None, overwrite_checkpoints: bool):
        if self.failure is not None:
            raise self.failure
        return argparse.Namespace(
            steps=max_steps or 1,
            initial_loss=1.0,
            final_loss=0.5,
            validation_loss=0.6,
            tokens_processed=32,
            elapsed_seconds=1.0,
        )

    def save_checkpoint(self, _path: Path) -> None:
        self.checkpoint.write_bytes(b"checkpoint-v1")


def _train_args(checkpoint: Path, *, resume: Path | None = None):
    return argparse.Namespace(
        command="train",
        config=Path("configs/tinyllm.yaml"),
        max_steps=1,
        resume=resume,
        checkpoint=checkpoint,
        overwrite=True,
        dry_run=False,
    )


def _install_cli_fakes(monkeypatch, cli_module, config, trainer):
    monkeypatch.setattr(cli_module, "_load_args_config", lambda _args: config)
    monkeypatch.setattr(
        cli_module, "_require_writable_output", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(cli_module, "_build_cli_trainer", lambda _config: (trainer, None))
    monkeypatch.setattr(run_manifest, "_git_revision", lambda: "abc1234")
    monkeypatch.setattr(run_manifest, "_runtime_metadata", lambda: {"python": "3.12.1"})
    monkeypatch.setattr(run_manifest, "_system_metadata", lambda: {"system": "Windows"})
    monkeypatch.setattr(run_manifest, "_accelerator_name", lambda: None)


def test_train_command_writes_completed_manifest_and_resume_keeps_run_identity(
    monkeypatch, tmp_path
):
    pytest.importorskip("torch")
    from tinyllm import cli

    config = _config()
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"run_dir": tmp_path / "runs"})}
    )
    checkpoint = tmp_path / "latest.pt"
    trainer = _FakeTrainer(checkpoint)
    _install_cli_fakes(monkeypatch, cli, config, trainer)

    cli._train_command(_train_args(checkpoint))
    manifest_path = tmp_path / "runs" / "run.json"
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first["status"] == "completed"
    assert first["finished_at"] is not None
    assert first["artifacts"]["arguments.checkpoint"]

    cli._train_command(_train_args(checkpoint, resume=checkpoint))
    resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed["run_id"] == first["run_id"]
    assert resumed["started_at"] == first["started_at"]
    assert resumed["status"] == "completed"
    assert resumed["resume_history"][-1]["started_at"]


def test_train_command_marks_manifest_failed_when_training_raises(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from tinyllm import cli

    config = _config()
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"run_dir": tmp_path / "runs"})}
    )
    checkpoint = tmp_path / "latest.pt"
    trainer = _FakeTrainer(checkpoint, failure=RuntimeError("training failed"))
    _install_cli_fakes(monkeypatch, cli, config, trainer)

    with pytest.raises(RuntimeError, match="training failed"):
        cli._train_command(_train_args(checkpoint))

    manifest = json.loads((tmp_path / "runs" / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["finished_at"] is not None


def test_train_command_marks_manifest_interrupted_on_keyboard_interrupt(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from tinyllm import cli

    config = _config()
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"run_dir": tmp_path / "runs"})}
    )
    trainer = _FakeTrainer(tmp_path / "latest.pt", failure=KeyboardInterrupt())
    _install_cli_fakes(monkeypatch, cli, config, trainer)

    with pytest.raises(KeyboardInterrupt):
        cli._train_command(_train_args(tmp_path / "latest.pt"))

    manifest = json.loads((tmp_path / "runs" / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["finished_at"] is not None


def test_train_command_links_metric_records_to_run_manifest(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from tinyllm import cli

    config = _config()
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"run_dir": tmp_path / "runs"})}
    )

    class MetricTrainer(cli.Trainer):
        def __init__(self):
            self.metric_logger = None

        def train(self, *, max_steps: int | None, overwrite_checkpoints: bool):
            self.metric_logger.log(1, {"loss": 0.5})
            return argparse.Namespace(
                steps=1,
                initial_loss=1.0,
                final_loss=0.5,
                validation_loss=None,
                tokens_processed=32,
                elapsed_seconds=1.0,
            )

        def save_checkpoint(self, path: Path) -> None:
            path.write_bytes(b"checkpoint-v1")

    trainer = MetricTrainer()
    _install_cli_fakes(monkeypatch, cli, config, trainer)

    cli._train_command(_train_args(tmp_path / "latest.pt"))

    manifest = json.loads((tmp_path / "runs" / "run.json").read_text(encoding="utf-8"))
    metric = json.loads(
        (tmp_path / "runs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert metric["run_id"] == manifest["run_id"]


def test_resume_restores_matching_archived_manifest(monkeypatch, tmp_path):
    config = _config()
    checkpoint_dir = tmp_path / "checkpoints"
    config = config.model_copy(
        update={
            "logging": config.logging.model_copy(update={"run_dir": tmp_path / "runs"}),
            "training": config.training.model_copy(update={"checkpoint_dir": checkpoint_dir}),
        }
    )
    monkeypatch.setattr(run_manifest, "_git_revision", lambda: "abc1234")
    monkeypatch.setattr(run_manifest, "_runtime_metadata", lambda: {"python": "3.12.1"})
    monkeypatch.setattr(run_manifest, "_system_metadata", lambda: {"system": "Windows"})
    monkeypatch.setattr(run_manifest, "_accelerator_name", lambda: None)

    first_checkpoint = checkpoint_dir / "first.pt"
    first_args = argparse.Namespace(command="train", checkpoint=first_checkpoint, resume=None)
    first_path, first = run_manifest.start_run_manifest(
        config, first_args, started_at="2026-10-28T12:00:00Z"
    )
    first_checkpoint.parent.mkdir(parents=True)
    first_checkpoint.write_bytes(b"first-checkpoint")
    run_manifest.finish_run_manifest(
        first_path,
        first,
        config,
        first_args,
        status="completed",
        finished_at="2026-10-28T12:01:00Z",
    )

    second_checkpoint = checkpoint_dir / "second.pt"
    second_args = argparse.Namespace(command="train", checkpoint=second_checkpoint, resume=None)
    second_path, second = run_manifest.start_run_manifest(
        config, second_args, started_at="2026-10-28T12:02:00Z"
    )
    second_checkpoint.write_bytes(b"second-checkpoint")
    run_manifest.finish_run_manifest(
        second_path,
        second,
        config,
        second_args,
        status="completed",
        finished_at="2026-10-28T12:03:00Z",
    )

    resume_args = argparse.Namespace(
        command="train", checkpoint=second_checkpoint, resume=first_checkpoint
    )
    _, resumed = run_manifest.start_run_manifest(
        config, resume_args, started_at="2026-10-28T12:04:00Z"
    )

    assert resumed["run_id"] == first["run_id"]
    assert resumed["config"] == first["config"]
    assert resumed["resume_history"][-1]["started_at"] == "2026-10-28T12:04:00Z"


def test_resume_does_not_reuse_running_manifest_for_unmatched_checkpoint(tmp_path):
    config = _config()
    checkpoint_dir = tmp_path / "checkpoints"
    config = config.model_copy(
        update={
            "logging": config.logging.model_copy(update={"run_dir": tmp_path / "runs"}),
            "training": config.training.model_copy(update={"checkpoint_dir": checkpoint_dir}),
        }
    )
    initial_args = argparse.Namespace(command="train", resume=None)
    _, initial = run_manifest.start_run_manifest(
        config, initial_args, started_at="2026-10-28T12:00:00Z"
    )
    unrelated_checkpoint = checkpoint_dir / "stale.pt"
    unrelated_checkpoint.parent.mkdir(parents=True)
    unrelated_checkpoint.write_bytes(b"checkpoint from another run")

    _, resumed = run_manifest.start_run_manifest(
        config,
        argparse.Namespace(command="train", resume=unrelated_checkpoint),
        started_at="2026-10-28T12:05:00Z",
    )

    assert resumed["run_id"] != initial["run_id"]


def test_resume_does_not_match_non_checkpoint_artifact_digest(tmp_path):
    config = _config()
    run_dir = tmp_path / "runs"
    config = config.model_copy(
        update={"logging": config.logging.model_copy(update={"run_dir": run_dir})}
    )
    manifest_path, initial = run_manifest.start_run_manifest(
        config,
        argparse.Namespace(command="train", resume=None),
        started_at="2026-10-28T12:00:00Z",
    )
    checkpoint = tmp_path / "stale.pt"
    checkpoint.write_bytes(b"same bytes as a non-checkpoint artifact")
    initial["status"] = "completed"
    initial["artifacts"] = {
        "data.train_tokens": run_manifest._file_sha256(checkpoint)
    }
    run_manifest.write_run_manifest(manifest_path, initial)

    _, resumed = run_manifest.start_run_manifest(
        config,
        argparse.Namespace(command="train", resume=checkpoint),
        started_at="2026-10-28T12:05:00Z",
    )

    assert resumed["run_id"] != initial["run_id"]
