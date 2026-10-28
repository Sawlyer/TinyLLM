import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tinyllm.training.evaluation import evaluate_model, write_evaluation_report
from tinyllm.cli import _build_parser


class PositionDataset:
    split = "validation"
    sequence_length = 3

    def __len__(self):
        return 5

    def batch(self, positions):
        values = (positions + 1).unsqueeze(1).expand(-1, self.sequence_length)
        return torch.zeros_like(values), values


class MeanTargetModel(torch.nn.Module):
    def forward(self, _inputs, targets):
        return SimpleNamespace(loss=targets.float().mean())


def test_evaluation_uses_seeded_positions_and_reports_batch_protocol():
    model = MeanTargetModel()
    model.train()
    dataset = PositionDataset()
    generator = torch.Generator().manual_seed(123)
    dataset.generator = generator
    original_state = generator.get_state().clone()

    first = evaluate_model(
        model,
        dataset,
        split="validation",
        seed=17,
        batch_size=2,
        batches=2,
        checkpoint_sha256="checkpoint-digest",
        corpus_sha256="validation-digest",
    )
    second = evaluate_model(
        model,
        dataset,
        split="validation",
        seed=17,
        batch_size=2,
        batches=2,
        checkpoint_sha256="checkpoint-digest",
        corpus_sha256="validation-digest",
    )

    assert first.mean_loss == second.mean_loss
    assert first.batch_loss_std == second.batch_loss_std
    assert first.evaluated_sequences == 4
    assert first.evaluated_tokens == 12
    assert first.batches == 2
    assert first.seed == 17
    assert first.split == "validation"
    assert first.checkpoint_sha256 == "checkpoint-digest"
    assert first.corpus_sha256 == "validation-digest"
    assert generator.get_state().equal(original_state)
    assert model.training


def test_token_limit_counts_exact_scored_tokens_and_weights_partial_batch():
    report = evaluate_model(
        MeanTargetModel(),
        PositionDataset(),
        split="validation",
        seed=9,
        batch_size=2,
        tokens=7,
    )
    generator = torch.Generator().manual_seed(9)
    first_positions = torch.randint(0, 5, (2,), generator=generator)
    last_position = torch.randint(0, 5, (1,), generator=generator)
    positions = torch.cat((first_positions, last_position))
    values = [float(position + 1) for position in positions]
    expected_mean = (values[0] * 3 + values[1] * 3 + values[2]) / 7
    first_batch_mean = (values[0] + values[1]) / 2
    expected_std = abs(first_batch_mean - values[2]) / 2

    assert report.requested_tokens == 7
    assert report.evaluated_tokens == 7
    assert report.evaluated_sequences == 3
    assert report.batches == 2
    assert report.mean_loss == pytest.approx(expected_mean)
    assert report.batch_loss_std == pytest.approx(expected_std)
    assert report.perplexity == pytest.approx(math.exp(expected_mean))


def test_evaluation_amount_must_select_positive_batches_or_tokens():
    arguments = {
        "model": MeanTargetModel(),
        "dataset": PositionDataset(),
        "split": "validation",
        "seed": 1,
        "batch_size": 2,
    }
    with pytest.raises(ValueError, match="exactly one"):
        evaluate_model(**arguments, batches=1, tokens=7)
    with pytest.raises(ValueError, match="positive"):
        evaluate_model(**arguments, batches=0)


def test_evaluation_rejects_mismatched_dataset_split():
    dataset = PositionDataset()
    dataset.split = "train"
    with pytest.raises(ValueError, match="does not match"):
        evaluate_model(
            MeanTargetModel(),
            dataset,
            split="validation",
            seed=1,
            batch_size=2,
            batches=1,
        )


def test_evaluate_cli_accepts_split_seed_token_limit_and_output_file():
    arguments = _build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            "checkpoint.pt",
            "--split",
            "train",
            "--seed",
            "17",
            "--tokens",
            "250",
            "--output",
            "evaluation.csv",
            "--overwrite",
        ]
    )

    assert arguments.split == "train"
    assert arguments.seed == 17
    assert arguments.tokens == 250
    assert arguments.batches == 8
    assert arguments.output == Path("evaluation.csv")
    assert arguments.overwrite

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["evaluate", "--checkpoint", "checkpoint.pt", "--batches", "2", "--tokens", "20"]
        )


def test_evaluate_command_uses_selected_split_and_writes_report(monkeypatch, tmp_path, capsys):
    from tinyllm import cli

    config = SimpleNamespace(
        training=SimpleNamespace(micro_batch_size=2, device="cpu"),
        model=object(),
    )
    manifest = SimpleNamespace(
        artifacts={"train": SimpleNamespace(sha256="train-corpus-sha256")}
    )
    dataset = PositionDataset()
    dataset.split = "train"
    monkeypatch.setattr(cli, "_load_args_config", lambda _args: config)
    monkeypatch.setattr(
        cli,
        "_validated_manifest",
        lambda _config, *, splits: (assert_split(splits, "train") or manifest),
    )
    monkeypatch.setattr(
        cli, "_dataset", lambda _config, _manifest, split, *, required: dataset
    )
    monkeypatch.setattr(cli, "_resolve_inference_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(cli, "load_model_checkpoint", lambda *_args, **_kwargs: SimpleNamespace(
        model=MeanTargetModel(), identity=SimpleNamespace(model_config=config.model)
    ))
    monkeypatch.setattr(cli, "_require_model_config_match", lambda *_args: None)
    monkeypatch.setattr(cli, "sha256_file", lambda _path: "checkpoint-sha256")

    output = tmp_path / "train-evaluation.json"
    args = _build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            "checkpoint.pt",
            "--split",
            "train",
            "--tokens",
            "7",
            "--seed",
            "9",
            "--output",
            str(output),
        ]
    )
    cli._evaluate_command(args)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["split"] == "train"
    assert report["evaluated_tokens"] == 7
    assert report["checkpoint_sha256"] == "checkpoint-sha256"
    assert report["corpus_sha256"] == "train-corpus-sha256"
    captured = capsys.readouterr()
    assert "nats/token" in captured.err
    assert json.loads(captured.out)["validation_loss"] is None


def test_evaluate_command_never_overwrites_its_checkpoint(tmp_path):
    from tinyllm import cli

    checkpoint = tmp_path / "checkpoint.pt"
    args = SimpleNamespace(output=checkpoint, checkpoint=checkpoint)
    with pytest.raises(ValueError, match="cannot replace"):
        cli._evaluate_command(args)


def assert_split(splits, expected):
    assert splits == {expected}


def test_report_writer_supports_json_and_csv_and_refuses_overwrite(tmp_path: Path):
    report = evaluate_model(
        MeanTargetModel(),
        PositionDataset(),
        split="validation",
        seed=2,
        batch_size=1,
        batches=1,
        checkpoint_sha256="checkpoint",
        corpus_sha256="corpus",
    )
    json_path = tmp_path / "evaluation.json"
    csv_path = tmp_path / "evaluation.csv"

    write_evaluation_report(report, json_path)
    write_evaluation_report(report, csv_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert saved["split"] == "validation"
    assert saved["evaluated_tokens"] == 3
    assert row["seed"] == "2"
    assert row["evaluated_tokens"] == "3"
    with pytest.raises(FileExistsError):
        write_evaluation_report(report, json_path)
    write_evaluation_report(report, json_path, overwrite=True)
    with pytest.raises(ValueError, match=r"\.json or \.csv"):
        write_evaluation_report(report, tmp_path / "evaluation.txt")
