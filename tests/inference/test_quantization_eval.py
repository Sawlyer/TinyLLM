import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import torch

from tinyllm.inference.quantization_eval import compare_quantized_models


class FixedLossModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, _inputs, _targets):
        return SimpleNamespace(loss=torch.tensor(self.value))


class ValidationDataset:
    split = "validation"
    sequence_length = 3

    def __len__(self):
        return 5

    def batch(self, positions):
        values = positions.unsqueeze(1).expand(-1, self.sequence_length)
        return torch.zeros_like(values), values


def test_comparison_reports_quality_cost_and_quantization_provenance():
    report = compare_quantized_models(
        FixedLossModel(1.0),
        FixedLossModel(1.25),
        ValidationDataset(),
        split="validation",
        seed=19,
        batch_size=2,
        batches=2,
        reference_checkpoint_sha256="a" * 64,
        quantized_checkpoint_sha256="b" * 64,
        corpus_sha256="c" * 64,
        reference_size_bytes=1000,
        quantized_size_bytes=600,
        quantized_manifest={
            "recipe": "int4-weight-only",
            "torch_version": "2.6.0",
            "torchao_version": "0.18.0",
            "eligible_linear_count": 4,
            "converted_linear_count": 3,
            "unconverted_linear_names": ["output"],
        },
        device="cpu",
        warmup_runs=0,
        measured_runs=2,
    ).to_dict()

    assert report["schema_version"] == 1
    assert report["evaluation"]["split"] == "validation"
    assert report["evaluation"]["seed"] == 19
    assert report["evaluation"]["evaluated_tokens"] == 12
    assert report["reference"]["mean_loss"] == 1.0
    assert report["quantized"]["mean_loss"] == 1.25
    assert report["quality"]["loss_delta_nats_per_token"] == 0.25
    assert report["quality"]["perplexity_ratio"] > 1.0
    assert report["reference"]["checkpoint_sha256"] == "a" * 64
    assert report["quantized"]["checkpoint_sha256"] == "b" * 64
    assert report["reference"]["artifact_size_bytes"] == 1000
    assert report["quantized"]["artifact_size_bytes"] == 600
    assert report["quantized"]["manifest"]["torchao_version"] == "0.18.0"
    assert report["quantized"]["manifest"]["unconverted_linear_names"] == ["output"]
    assert report["reference"]["latency_ms_per_token"] >= 0
    assert report["quantized"]["latency_ms_per_token"] >= 0
    assert report["reference"]["peak_cuda_memory_bytes"] is None
    assert report["quantized"]["peak_cuda_memory_bytes"] is None


def test_compare_quantization_cli_accepts_reproducible_measurement_options(tmp_path):
    from tinyllm.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "compare-quantization",
            "--checkpoint",
            str(tmp_path / "source.pt"),
            "--quantized",
            str(tmp_path / "int4.pt"),
            "--tokens",
            "120",
            "--seed",
            "7",
            "--warmup-runs",
            "0",
            "--measured-runs",
            "4",
        ]
    )

    assert args.split == "validation"
    assert args.tokens == 120
    assert args.seed == 7
    assert args.warmup_runs == 0
    assert args.measured_runs == 4
    assert args.handler.__name__ == "_compare_quantization_command"


def test_compare_quantization_command_loads_matching_artifacts_and_writes_report(
    monkeypatch, tmp_path, capsys
):
    from tinyllm import cli

    source_path = tmp_path / "source.pt"
    quantized_path = tmp_path / "quantized.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    output_path = tmp_path / "comparison.json"
    source_path.write_bytes(b"source")
    quantized_path.write_bytes(b"quantized")
    tokenizer_path.write_bytes(b"tokenizer")
    tokenizer_sha = "d" * 64
    model_config = object()
    dataset = ValidationDataset()
    config = SimpleNamespace(
        training=SimpleNamespace(micro_batch_size=2, device="cpu"),
        model=model_config,
        tokenizer=SimpleNamespace(path=tokenizer_path),
    )
    corpus_manifest = SimpleNamespace(
        artifacts={"validation": SimpleNamespace(sha256="e" * 64)}
    )
    source = SimpleNamespace(
        model=FixedLossModel(1.0),
        identity=SimpleNamespace(model_config=model_config, tokenizer_sha256=tokenizer_sha),
    )
    quantized = SimpleNamespace(
        model=FixedLossModel(1.1),
        model_config=SimpleNamespace(model_dump=lambda mode: model_config),
        manifest=SimpleNamespace(
            tokenizer_sha256=tokenizer_sha,
            to_dict=lambda: {"recipe": "int4-weight-only"},
        ),
    )
    report = SimpleNamespace(
        quality={
            "loss_delta_nats_per_token": 0.1,
            "perplexity_ratio": 1.1,
            "latency_ratio": 0.8,
            "artifact_size_ratio": 0.5,
        },
        evaluation={"evaluated_tokens": 12},
        to_dict=lambda: {"schema_version": 1},
    )
    monkeypatch.setattr(cli, "_load_args_config", lambda _args: config)
    monkeypatch.setattr(
        cli,
        "_validated_manifest",
        lambda _config, *, splits: corpus_manifest,
    )
    monkeypatch.setattr(
        cli, "_dataset", lambda *_args, **_kwargs: dataset
    )
    monkeypatch.setattr(cli, "_resolve_inference_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(cli, "load_model_checkpoint", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(cli, "load_quantized_checkpoint", lambda *_args, **_kwargs: quantized)
    monkeypatch.setattr(cli, "_require_model_config_match", lambda *_args: None)
    monkeypatch.setattr(
        cli,
        "sha256_file",
        lambda path: tokenizer_sha if Path(path) == tokenizer_path else "f" * 64,
    )
    observed = {}

    def compare(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return report

    monkeypatch.setattr(cli, "compare_quantized_models", compare)
    monkeypatch.setattr(
        cli,
        "write_quantization_comparison_report",
        lambda *args, **kwargs: output_path.write_text(json.dumps(report.to_dict())),
    )
    args = SimpleNamespace(
        checkpoint=source_path,
        quantized=quantized_path,
        output=output_path,
        config=None,
        seed=42,
        tokens=12,
        batches=None,
        split="validation",
        device=None,
        overwrite=False,
        warmup_runs=0,
        measured_runs=2,
    )

    cli._compare_quantization_command(args)

    assert observed["args"][:3] == (source.model, quantized.model, dataset)
    assert observed["kwargs"]["tokens"] == 12
    assert observed["kwargs"]["seed"] == 42
    assert observed["kwargs"]["quantized_manifest"] == {"recipe": "int4-weight-only"}
    assert json.loads(output_path.read_text()) == {"schema_version": 1}
    assert json.loads(capsys.readouterr().out)["evaluated_tokens"] == 12


def test_comparison_rejects_empty_measurement_runs():
    with pytest.raises(ValueError, match="measured_runs"):
        compare_quantized_models(
            FixedLossModel(1),
            FixedLossModel(1),
            ValidationDataset(),
            split="validation",
            seed=1,
            batch_size=1,
            batches=1,
            reference_checkpoint_sha256="a" * 64,
            quantized_checkpoint_sha256="b" * 64,
            corpus_sha256="c" * 64,
            reference_size_bytes=1,
            quantized_size_bytes=1,
            quantized_manifest={},
            device="cpu",
            measured_runs=0,
        )

