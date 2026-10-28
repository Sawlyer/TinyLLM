import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tinyllm.cli import _build_training_preflight_report, main
from tinyllm.config.load import load_config
from tinyllm.config.schema import TinyLLMConfig
from tinyllm.model.transformer import TinyLLM, count_parameters, count_parameters_for_config


def _tiny_config(tmp_path: Path, *, device: str = "cpu", precision: str = "fp32") -> TinyLLMConfig:
    root = Path(__file__).parents[2]
    raw = load_config(root / "configs" / "tinyllm.yaml", []).model_dump(mode="python")
    raw["data"].update(
        {
            "cache_dir": tmp_path / "cache",
            "output_dir": tmp_path / "processed",
            "validation_fraction": 0.5,
        }
    )
    raw["tokenizer"].update(
        {
            "vocab_size": 128,
            "min_frequency": 1,
            "path": tmp_path / "processed" / "tokenizer.json",
        }
    )
    raw["model"].update(
        {
            "vocab_size": 128,
            "max_seq_len": 8,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "n_kv_heads": 1,
            "mlp_ratio": 2.0,
        }
    )
    raw["training"].update(
        {
            "device": device,
            "precision": precision,
            "max_steps": 4,
            "warmup_steps": 1,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "checkpoint_dir": tmp_path / "checkpoints",
        }
    )
    raw["logging"].update({"run_dir": tmp_path / "runs", "checkpoint_interval": 2})
    return TinyLLMConfig.model_validate(raw)


def _args(
    tmp_path: Path,
    *,
    overwrite: bool = False,
    resume: Path | None = None,
    max_steps: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=tmp_path / "checkpoints" / "latest.pt",
        resume=resume,
        overwrite=overwrite,
        max_steps=max_steps,
    )


def test_parameter_count_from_config_matches_model_without_duplicate_tied_weights(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    model = TinyLLM(config.model)

    assert count_parameters_for_config(config.model) == count_parameters(model)


def test_dry_run_does_not_construct_model_or_write_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _tiny_config(tmp_path)
    model_config = tmp_path / "tinyllm.yaml"
    model_config.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )

    def fail_if_constructed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not instantiate the model")

    monkeypatch.setattr("tinyllm.cli.TinyLLM", fail_if_constructed)
    main(["train", "--config", str(model_config), "--dry-run"])

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "warning"
    assert report["model"]["parameter_count"] == count_parameters_for_config(config.model)
    assert not (tmp_path / "checkpoints").exists()
    assert not (tmp_path / "runs").exists()


def test_preflight_reports_unavailable_explicit_cuda_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config(tmp_path, device="cuda:3")
    monkeypatch.setattr("tinyllm.training.trainer.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("tinyllm.training.trainer.torch.cuda.device_count", lambda: 1)

    report = _build_training_preflight_report(config, _args(tmp_path), require_data=False)

    assert report["status"] == "error"
    assert any("cuda:3" in item["cause"] for item in report["errors"])
    assert any("available CUDA device" in item["fix"] for item in report["errors"])


def test_preflight_reports_existing_checkpoint_without_overwrite(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"existing")

    report = _build_training_preflight_report(config, _args(tmp_path), require_data=False)

    assert any(item["field"] == "checkpoint" for item in report["errors"])


def test_preflight_reports_insufficient_disk_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config(tmp_path)
    monkeypatch.setattr(
        "tinyllm.cli.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=1000, used=1000, free=0),
    )

    report = _build_training_preflight_report(config, _args(tmp_path), require_data=False)

    assert report["status"] == "error"
    assert any(item["field"] == "storage" for item in report["errors"])


def test_preflight_rejects_zero_max_steps_instead_of_using_config_default(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)

    report = _build_training_preflight_report(
        config, _args(tmp_path, max_steps=0), require_data=False
    )

    assert report["status"] == "error"
    assert any(item["field"] == "training.max_steps" for item in report["errors"])


def test_preflight_rejects_corrupt_resume_before_writing_run_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config(tmp_path)
    manifest_path = config.data.output_dir / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")
    resume = tmp_path / "corrupt.pt"
    resume.write_bytes(b"corrupt")
    manifest = SimpleNamespace(
        artifacts={"train": SimpleNamespace(sha256="corpus-hash")},
        tokenizer_sha256="tokenizer-hash",
        dataset_name="example",
        dataset_revision="revision",
    )
    train_dataset = SimpleNamespace(
        sequence_length=8,
        validate_state_dict=lambda _state: None,
    )
    monkeypatch.setattr("tinyllm.cli._validated_manifest", lambda _config: manifest)
    monkeypatch.setattr(
        "tinyllm.cli._dataset",
        lambda _config, _manifest, split, *, required: train_dataset if split == "train" else None,
    )
    monkeypatch.setattr(
        "tinyllm.cli.load_training_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid checkpoint bytes")),
    )

    report = _build_training_preflight_report(
        config, _args(tmp_path, resume=resume), require_data=True
    )

    assert report["status"] == "error"
    assert any(
        item["field"] == "resume" and "invalid checkpoint bytes" in item["cause"]
        for item in report["errors"]
    )


def test_train_preflight_errors_before_creating_run_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config(tmp_path, device="cuda")
    model_config = tmp_path / "tinyllm.yaml"
    model_config.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr("tinyllm.cli.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "tinyllm.cli.start_run_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight must run before writing the run manifest")
        ),
    )

    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(model_config)])

    assert error.value.code == 1
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "checkpoints").exists()

