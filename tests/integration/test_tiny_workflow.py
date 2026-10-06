from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import yaml

import tinyllm.cli as cli_module
from tinyllm.cli import main
from tinyllm.config.load import load_config


@dataclass(frozen=True)
class Invocation:
    exit_code: int
    stdout: str
    stderr: str


class CliRunner:
    def invoke(self, arguments: list[str]) -> Invocation:
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = 0
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                main(arguments)
            except SystemExit as error:
                exit_code = int(error.code or 0)
            except Exception as error:  # CLI failures are observable through exit status.
                exit_code = 1
                stderr.write(f"{type(error).__name__}: {error}\n")
        return Invocation(exit_code=exit_code, stdout=stdout.getvalue(), stderr=stderr.getvalue())


@dataclass(frozen=True)
class TinyTextSource:
    config: Path
    checkpoint: Path
    resumed_checkpoint: Path


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tiny_text_source(tmp_path: Path) -> TinyTextSource:
    source = tmp_path / "tiny.txt"
    source.write_text(
        "\n".join(
            f"Once a tiny model learned deterministic pattern number {index}."
            for index in range(80)
        ),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint = checkpoint_dir / "latest.pt"
    resumed_checkpoint = checkpoint_dir / "resumed.pt"
    config = tmp_path / "tiny.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset_name": str(source),
                    "dataset_revision": None,
                    "cache_dir": str(tmp_path / "cache"),
                    "output_dir": str(tmp_path / "processed"),
                    "text_field": "text",
                    "validation_fraction": 0.25,
                    "seed": 7,
                },
                "tokenizer": {
                    "vocab_size": 64,
                    "min_frequency": 1,
                    "special_tokens": ["<unk>", "<bos>", "<eos>", "<pad>"],
                    "path": str(tmp_path / "tokenizer.json"),
                },
                "model": {
                    "vocab_size": 64,
                    "max_seq_len": 8,
                    "d_model": 16,
                    "n_layers": 1,
                    "n_heads": 4,
                    "n_kv_heads": 2,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "rope_theta": 10000.0,
                    "rms_norm_eps": 1e-5,
                },
                "training": {
                    "seed": 7,
                    "device": "cpu",
                    "precision": "fp32",
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "max_steps": 13,
                    "learning_rate": 1e-3,
                    "min_learning_rate": 1e-4,
                    "warmup_steps": 1,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "compile": False,
                    "checkpoint_dir": str(checkpoint_dir),
                },
                "logging": {
                    "run_dir": str(tmp_path / "runs"),
                    "log_interval": 13,
                    "validation_interval": 13,
                    "checkpoint_interval": 100,
                },
                "generation": {
                    "max_new_tokens": 4,
                    "temperature": 0.0,
                    "top_k": None,
                    "top_p": 1.0,
                    "seed": 7,
                    "eos_token_id": None,
                },
                "quantization": {
                    "recipe": None,
                    "output_dir": str(tmp_path / "exports"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return TinyTextSource(config, checkpoint, resumed_checkpoint)


def test_tiny_corpus_can_prepare_train_resume_and_generate(
    cli_runner: CliRunner,
    tiny_text_source: TinyTextSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = cli_runner.invoke(["prepare", "--config", str(tiny_text_source.config)])
    assert prepared.exit_code == 0, prepared.stderr

    trained = cli_runner.invoke(
        ["train", "--config", str(tiny_text_source.config), "--max-steps", "12"]
    )
    assert trained.exit_code == 0, trained.stderr
    assert tiny_text_source.checkpoint.is_file()

    resumed = cli_runner.invoke(
        [
            "train",
            "--config",
            str(tiny_text_source.config),
            "--resume",
            str(tiny_text_source.checkpoint),
            "--checkpoint",
            str(tiny_text_source.resumed_checkpoint),
            "--max-steps",
            "13",
        ]
    )
    assert resumed.exit_code == 0, resumed.stderr

    generated = cli_runner.invoke(
        [
            "generate",
            "--checkpoint",
            str(tiny_text_source.resumed_checkpoint),
            "--prompt",
            "Once",
        ]
    )
    assert generated.exit_code == 0, generated.stderr
    assert len(generated.stdout.strip()) > len("Once")

    resolved_eos: list[int | None] = []

    def capture_eos(model, input_ids, config, *, use_cache):
        del model, use_cache
        resolved_eos.append(config.eos_token_id)
        return torch.cat([input_ids, input_ids.new_tensor([[config.eos_token_id]])], dim=1)

    monkeypatch.setattr("tinyllm.cli.generate", capture_eos)
    eos_lookup = cli_runner.invoke(
        ["generate", "--checkpoint", str(tiny_text_source.resumed_checkpoint), "--prompt", "Once"]
    )
    assert eos_lookup.exit_code == 0, eos_lookup.stderr
    assert resolved_eos and resolved_eos[0] is not None

    class MissingEosTokenizer:
        def encode(self, prompt: str):
            del prompt
            return type("Encoding", (), {"ids": [1]})()

        def token_to_id(self, token: str) -> None:
            del token
            return None

    monkeypatch.setattr("tinyllm.cli.Tokenizer.from_file", lambda path: MissingEosTokenizer())
    missing_eos = cli_runner.invoke(
        ["generate", "--checkpoint", str(tiny_text_source.resumed_checkpoint), "--prompt", "Once"]
    )
    assert missing_eos.exit_code != 0
    assert "required <eos> token" in missing_eos.stderr
    monkeypatch.undo()

    invalid_generation = cli_runner.invoke(
        [
            "generate",
            "--checkpoint",
            str(tiny_text_source.resumed_checkpoint),
            "--prompt",
            "Once",
            "--max-new-tokens",
            "-1",
        ]
    )
    assert invalid_generation.exit_code != 0
    assert "max_new_tokens" in invalid_generation.stderr

    inspected = cli_runner.invoke(["inspect-data", "--config", str(tiny_text_source.config)])
    assert inspected.exit_code == 0, inspected.stderr
    assert '"checksum_validated": true' in inspected.stdout

    evaluated = cli_runner.invoke(
        [
            "evaluate",
            "--config",
            str(tiny_text_source.config),
            "--checkpoint",
            str(tiny_text_source.resumed_checkpoint),
            "--batches",
            "2",
        ]
    )
    assert evaluated.exit_code == 0, evaluated.stderr
    assert '"validation_loss":' in evaluated.stdout

    benchmark_output = tiny_text_source.config.parent / "benchmark.json"
    benchmarked = cli_runner.invoke(
        [
            "benchmark-precision",
            "--config",
            str(tiny_text_source.config),
            "--modes",
            "fp32",
            "--warmup-steps",
            "0",
            "--measured-steps",
            "1",
            "--output",
            str(benchmark_output),
        ]
    )
    assert benchmarked.exit_code == 0, benchmarked.stderr
    assert benchmark_output.is_file()

    zero_batch = cli_runner.invoke(
        [
            "benchmark-precision",
            "--config",
            str(tiny_text_source.config),
            "--modes",
            "fp32",
            "--batch-size",
            "0",
        ]
    )
    assert zero_batch.exit_code != 0
    assert "--batch-size must be positive" in zero_batch.stderr

    quantized_output = tiny_text_source.config.parent / "tiny-bf16.pt"
    quantized = cli_runner.invoke(
        [
            "quantize",
            "--config",
            str(tiny_text_source.config),
            "--checkpoint",
            str(tiny_text_source.resumed_checkpoint),
            "--recipe",
            "bf16",
            "--output",
            str(quantized_output),
        ]
    )
    assert quantized.exit_code == 0, quantized.stderr
    assert quantized_output.is_file()

    reused = cli_runner.invoke(["prepare", "--config", str(tiny_text_source.config)])
    assert reused.exit_code == 0, reused.stderr
    assert '"dataset":' in reused.stdout


def test_benchmark_cli_exits_nonzero_after_reporting_failed_mode(
    cli_runner: CliRunner,
    tiny_text_source: TinyTextSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class FailedResult:
        mode: str = "bf16"
        status: str = "failed"
        reason: str = "RuntimeError: kernel exploded"

        def to_dict(self) -> dict[str, object]:
            return {"mode": self.mode, "status": self.status, "reason": self.reason}

    monkeypatch.setattr(
        cli_module,
        "benchmark_precision",
        lambda config, modes: [FailedResult()],
    )

    result = cli_runner.invoke(
        [
            "benchmark-precision",
            "--config",
            str(tiny_text_source.config),
            "--modes",
            "bf16",
        ]
    )

    assert result.exit_code == 1
    assert '"status": "failed"' in result.stdout
    assert "bf16: RuntimeError: kernel exploded" in result.stderr


def test_benchmark_cli_exits_zero_after_reporting_unsupported_mode(
    cli_runner: CliRunner,
    tiny_text_source: TinyTextSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass(frozen=True)
    class UnsupportedResult:
        mode: str = "fp8"
        status: str = "unsupported"
        reason: str = "TritonMissing: no working triton installation"

        def to_dict(self) -> dict[str, object]:
            return {"mode": self.mode, "status": self.status, "reason": self.reason}

    monkeypatch.setattr(
        cli_module,
        "benchmark_precision",
        lambda config, modes: [UnsupportedResult()],
    )

    result = cli_runner.invoke(
        [
            "benchmark-precision",
            "--config",
            str(tiny_text_source.config),
            "--modes",
            "fp8",
        ]
    )

    assert result.exit_code == 0, result.stderr
    assert '"status": "unsupported"' in result.stdout
    assert result.stderr == ""


def test_help_lists_complete_command_surface(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(["--help"])

    assert result.exit_code == 0
    for command in (
        "prepare",
        "inspect-data",
        "train",
        "evaluate",
        "generate",
        "benchmark-precision",
        "quantize",
    ):
        assert command in result.stdout


def test_malformed_tokenizer_is_reported_as_invalid_file(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{not valid tokenizer JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid tokenizer file"):
        cli_module._load_tokenizer(tokenizer_path)


def test_cli_trainer_normalizes_expected_cuda_error(
    tiny_text_source: TinyTextSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(tiny_text_source.config, [])
    message = "CUDA training requested but CUDA is unavailable"

    def fail_to_build_trainer(_config):
        raise RuntimeError(message)

    monkeypatch.setattr(cli_module, "_build_trainer", fail_to_build_trainer)

    with pytest.raises(ValueError, match=message):
        cli_module._build_cli_trainer(config)


def test_cli_trainer_preserves_unexpected_runtime_error(
    tiny_text_source: TinyTextSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(tiny_text_source.config, [])
    unexpected = RuntimeError("CUDA kernel launch failed")

    def fail_to_build_trainer(_config):
        raise unexpected

    monkeypatch.setattr(cli_module, "_build_trainer", fail_to_build_trainer)

    with pytest.raises(RuntimeError) as raised:
        cli_module._build_cli_trainer(config)

    assert raised.value is unexpected


def test_train_dry_run_validates_identity_and_applies_overrides(
    cli_runner: CliRunner,
    tiny_text_source: TinyTextSource,
) -> None:
    prepared = cli_runner.invoke(["prepare", "--config", str(tiny_text_source.config)])
    assert prepared.exit_code == 0, prepared.stderr

    result = cli_runner.invoke(
        [
            "train",
            "--config",
            str(tiny_text_source.config),
            "--set",
            "training.micro_batch_size=2",
            "--dry-run",
        ]
    )
    assert result.exit_code == 0, result.stderr
    assert "dataset identity: validated" in result.stdout
    assert "parameters:" in result.stdout
    assert "effective batch: 2 sequences" in result.stdout
    assert "precision: FP32" in result.stdout
    assert "device: cpu" in result.stdout
    assert not tiny_text_source.checkpoint.exists()


def test_config_must_match_prepared_manifest(
    cli_runner: CliRunner,
    tiny_text_source: TinyTextSource,
) -> None:
    prepared = cli_runner.invoke(["prepare", "--config", str(tiny_text_source.config)])
    assert prepared.exit_code == 0, prepared.stderr

    result = cli_runner.invoke(
        [
            "inspect-data",
            "--config",
            str(tiny_text_source.config),
            "--set",
            "tokenizer.min_frequency=2",
        ]
    )
    assert result.exit_code != 0
    assert "tokenizer configuration is incompatible" in result.stderr


def test_periodic_checkpoint_refuses_collision_without_overwrite(
    cli_runner: CliRunner,
    tiny_text_source: TinyTextSource,
) -> None:
    prepared = cli_runner.invoke(["prepare", "--config", str(tiny_text_source.config)])
    assert prepared.exit_code == 0, prepared.stderr
    periodic = tiny_text_source.checkpoint.parent / "step-00000001.pt"
    periodic.parent.mkdir(parents=True, exist_ok=True)
    periodic.write_bytes(b"keep")

    result = cli_runner.invoke(
        [
            "train",
            "--config",
            str(tiny_text_source.config),
            "--set",
            "logging.checkpoint_interval=1",
            "--max-steps",
            "1",
        ]
    )
    assert result.exit_code != 0
    assert "checkpoint already exists" in result.stderr
    assert periodic.read_bytes() == b"keep"

    replaced = cli_runner.invoke(
        [
            "train",
            "--config",
            str(tiny_text_source.config),
            "--set",
            "logging.checkpoint_interval=1",
            "--max-steps",
            "1",
            "--overwrite",
        ]
    )
    assert replaced.exit_code == 0, replaced.stderr
    assert periodic.read_bytes() != b"keep"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("int8", "int8-weight-only"),
        ("int4", "int4-weight-only"),
    ],
)
def test_quantize_maps_short_weight_only_aliases_to_api_recipes(
    alias: str,
    canonical: str,
    cli_runner: CliRunner,
    tiny_text_source: TinyTextSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tiny_text_source.config.parent / f"tiny-{alias}.pt"
    recorded_recipes: list[str] = []

    class FakeManifest:
        def __init__(self, recipe: str) -> None:
            self.recipe = recipe

        def to_dict(self) -> dict[str, str]:
            return {"recipe": self.recipe}

    def fake_quantize_checkpoint(
        checkpoint: Path,
        destination: Path,
        recipe: str,
        *,
        overwrite: bool,
        device: str,
    ) -> FakeManifest:
        assert checkpoint == tiny_text_source.resumed_checkpoint
        assert destination == output
        assert overwrite is False
        assert device == "cpu"
        recorded_recipes.append(recipe)
        return FakeManifest(recipe)

    monkeypatch.setattr("tinyllm.cli.quantize_checkpoint", fake_quantize_checkpoint)
    result = cli_runner.invoke(
        [
            "quantize",
            "--config",
            str(tiny_text_source.config),
            "--checkpoint",
            str(tiny_text_source.resumed_checkpoint),
            "--recipe",
            alias,
            "--output",
            str(output),
        ]
    )
    assert result.exit_code == 0, result.stderr
    assert recorded_recipes == [canonical]
