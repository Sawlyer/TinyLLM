import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tinyllm.config.load import load_config
from tinyllm.config.schema import TinyLLMConfig
from tinyllm.data.artifacts import MANIFEST_VERSION, CorpusManifest, write_token_artifact
from tinyllm.data.loader import TokenDataset
from tinyllm.model.transformer import TinyLLM
from tinyllm.training.metrics import MetricLogger
from tinyllm.training.trainer import Trainer

TOKENIZER_SHA256 = "a" * 64


def _config(
    tmp_path: Path,
    *,
    device: str = "cpu",
    micro_batch_size: int = 4,
    accumulation: int = 2,
    max_grad_norm: float = 1.0,
    n_layers: int = 1,
    log_interval: int = 5,
) -> TinyLLMConfig:
    return load_config(
        Path("configs/tinyllm.yaml"),
        [
            "tokenizer.vocab_size=8",
            "model.vocab_size=8",
            "model.max_seq_len=4",
            "model.d_model=8",
            f"model.n_layers={n_layers}",
            "model.n_heads=2",
            "model.n_kv_heads=1",
            "model.mlp_ratio=2.0",
            f"training.device={device}",
            "training.precision=fp32",
            f"training.micro_batch_size={micro_batch_size}",
            f"training.gradient_accumulation_steps={accumulation}",
            "training.max_steps=30",
            "training.learning_rate=0.03",
            "training.min_learning_rate=0.003",
            "training.warmup_steps=0",
            "training.weight_decay=0.0",
            f"training.max_grad_norm={max_grad_norm}",
            f"training.checkpoint_dir={tmp_path / 'checkpoints'}",
            f"logging.run_dir={tmp_path / 'runs'}",
            f"logging.log_interval={log_interval}",
            "logging.validation_interval=5",
        ],
    )


def _datasets(tmp_path: Path, *, seed: int = 7) -> tuple[TokenDataset, TokenDataset]:
    output_dir = tmp_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = np.tile(np.arange(8, dtype=np.uint16), 64)
    artifacts = {
        split: write_token_artifact(
            output_dir,
            split,
            pattern,
            tokenizer_sha256=TOKENIZER_SHA256,
            overwrite=True,
        )
        for split in ("train", "validation")
    }
    manifest = CorpusManifest(
        version=MANIFEST_VERSION,
        dataset_name="training-fixture",
        dataset_requested_revision=None,
        dataset_revision="injected-sha256:" + "b" * 64,
        dataset_fingerprint="training-fixture",
        text_field="text",
        seed=seed,
        validation_fraction=0.5,
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        tokenizer_sha256=TOKENIZER_SHA256,
        tokenizer_config={"vocab_size": 8},
        data_config={
            "dataset_name": "training-fixture",
            "dataset_revision": None,
            "text_field": "text",
            "seed": seed,
            "validation_fraction": 0.5,
            "output_dir": str(output_dir),
        },
        document_counts={"train": 1, "validation": 1},
        artifacts=artifacts,
        tokenizer_vocab_size=8,
        special_token_ids={"<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3},
    )
    return (
        TokenDataset(manifest, "train", 4, seed, artifact_dir=output_dir),
        TokenDataset(manifest, "validation", 4, seed + 1, artifact_dir=output_dir),
    )


@pytest.fixture
def tiny_training_fixture(tmp_path: Path) -> Trainer:
    config = _config(tmp_path)
    train_dataset, validation_dataset = _datasets(tmp_path)
    torch.manual_seed(11)
    return Trainer(config, TinyLLM(config.model), train_dataset, validation_dataset)


def test_tiny_training_reduces_loss(tiny_training_fixture: Trainer) -> None:
    summary = tiny_training_fixture.train(max_steps=30)

    assert summary.steps == 30
    assert summary.final_loss < summary.initial_loss
    assert summary.validation_loss is not None
    assert math.isfinite(summary.validation_loss)


def test_training_writes_jsonl_and_tensorboard_metrics(
    tiny_training_fixture: Trainer,
) -> None:
    tiny_training_fixture.train(max_steps=2)
    run_dir = tiny_training_fixture.config.logging.run_dir

    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert records[-1]["step"] == 2
    assert records[-1]["tokens_processed"] > 0
    assert records[-1]["learning_rate"] > 0
    assert list((run_dir / "tensorboard").glob("events.out.tfevents.*"))


def test_gradient_accumulation_is_normalized_to_effective_batch(tmp_path: Path) -> None:
    class GradientCaptureOptimizer(torch.optim.Optimizer):
        def __init__(self, parameters) -> None:
            super().__init__(parameters, defaults={})
            self.gradients: list[torch.Tensor] = []

        def step(self, closure=None):
            self.gradients = [
                parameter.grad.detach().clone()
                for group in self.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]

    accumulated_config = _config(
        tmp_path / "accumulated",
        micro_batch_size=2,
        accumulation=2,
        max_grad_norm=1e9,
    )
    direct_config = _config(
        tmp_path / "direct",
        micro_batch_size=4,
        accumulation=1,
        max_grad_norm=1e9,
    )
    accumulated_data, _ = _datasets(tmp_path / "accumulated-data", seed=19)
    direct_data, _ = _datasets(tmp_path / "direct-data", seed=19)
    torch.manual_seed(23)
    accumulated_model = TinyLLM(accumulated_config.model)
    direct_model = TinyLLM(direct_config.model)
    direct_model.load_state_dict(accumulated_model.state_dict())
    accumulated_optimizer = GradientCaptureOptimizer(accumulated_model.parameters())
    direct_optimizer = GradientCaptureOptimizer(direct_model.parameters())

    Trainer(
        accumulated_config,
        accumulated_model,
        accumulated_data,
        optimizer=accumulated_optimizer,
    ).train(max_steps=1)
    Trainer(
        direct_config,
        direct_model,
        direct_data,
        optimizer=direct_optimizer,
    ).train(max_steps=1)

    for accumulated, direct in zip(
        accumulated_optimizer.gradients, direct_optimizer.gradients, strict=True
    ):
        assert torch.allclose(accumulated, direct, atol=1e-6, rtol=1e-5)


def test_non_finite_loss_stops_training_with_actionable_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    train_dataset, _ = _datasets(tmp_path)
    model = TinyLLM(config.model)
    with torch.no_grad():
        model.token_embedding.weight.fill_(float("nan"))
    logger = MetricLogger(config.logging.run_dir)

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        Trainer(config, model, train_dataset, metric_logger=logger).train(max_steps=1)

    _assert_diagnostic_checkpoint(config, "non-finite loss")
    with pytest.raises(RuntimeError, match="closed"):
        logger.log(0, {})


def test_evaluation_restores_training_mode_when_forward_fails(tmp_path: Path) -> None:
    class FailingTinyLLM(TinyLLM):
        def forward(self, *_: object, **__: object):
            raise RuntimeError("evaluation failure")

    config = _config(tmp_path)
    train_dataset, _ = _datasets(tmp_path)
    model = FailingTinyLLM(config.model)

    with pytest.raises(RuntimeError, match="evaluation failure"):
        Trainer(config, model, train_dataset).train(max_steps=1)

    assert model.training


def test_microbatch_accumulation_does_not_materialize_device_scalars(
    tmp_path: Path, monkeypatch
) -> None:
    original_float = torch.Tensor.__float__
    original_bool = torch.Tensor.__bool__
    counts = {"float": 0, "bool": 0}

    def counted_float(tensor: torch.Tensor) -> float:
        counts["float"] += 1
        return original_float(tensor)

    def counted_bool(tensor: torch.Tensor) -> bool:
        counts["bool"] += 1
        return original_bool(tensor)

    monkeypatch.setattr(torch.Tensor, "__float__", counted_float)
    monkeypatch.setattr(torch.Tensor, "__bool__", counted_bool)

    def materializations(
        name: str, *, micro_batch_size: int, accumulation: int, n_layers: int = 1
    ) -> dict:
        counts.update(float=0, bool=0)
        config = _config(
            tmp_path / name,
            micro_batch_size=micro_batch_size,
            accumulation=accumulation,
            n_layers=n_layers,
        )
        train_dataset, _ = _datasets(tmp_path / f"{name}-data", seed=31)
        torch.manual_seed(37)
        Trainer(config, TinyLLM(config.model), train_dataset).train(max_steps=1)
        return counts.copy()

    one_microstep = materializations("one", micro_batch_size=4, accumulation=1)
    four_microsteps = materializations("four", micro_batch_size=1, accumulation=4)
    two_layers = materializations("two-layers", micro_batch_size=4, accumulation=1, n_layers=2)

    assert four_microsteps == one_microstep
    assert two_layers == one_microstep


def test_throughput_excludes_evaluation_validation_and_metric_flush(
    tmp_path: Path,
) -> None:
    class ManualClock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    class AdvancingSGD(torch.optim.SGD):
        def step(self, closure=None):
            clock.advance(2.0)
            return super().step(closure)

    class AdvancingMetricLogger:
        def __init__(self) -> None:
            self.records: list[dict[str, int | float]] = []

        def log(self, step: int, metrics: dict[str, int | float]) -> None:
            self.records.append({"step": step, **metrics})
            clock.advance(300.0)

        def close(self) -> None:
            pass

    class TimedTrainer(Trainer):
        def _monitor_loss(self, dataset: TokenDataset) -> float:
            clock.advance(100.0)
            return super()._monitor_loss(dataset)

        def _validation_loss(self, dataset: TokenDataset | None) -> float:
            clock.advance(200.0)
            return super()._validation_loss(dataset)

    clock = ManualClock()
    config = _config(tmp_path, log_interval=1)
    train_dataset, validation_dataset = _datasets(tmp_path)
    model = TinyLLM(config.model)
    optimizer = AdvancingSGD(model.parameters(), lr=0.01)
    logger = AdvancingMetricLogger()

    TimedTrainer(
        config,
        model,
        train_dataset,
        validation_dataset,
        optimizer=optimizer,
        metric_logger=logger,
        clock=clock,
    ).train(max_steps=2)

    assert [record["tokens_per_second"] for record in logger.records] == pytest.approx([16.0, 16.0])


def test_trainer_consumes_prefetched_batches_instead_of_direct_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    train_dataset, _ = _datasets(tmp_path)
    trainer = Trainer(config, TinyLLM(config.model), train_dataset)

    def direct_sampling_is_forbidden(batch_size: int) -> torch.Tensor:
        raise AssertionError(f"direct sample_positions call for batch size {batch_size}")

    monkeypatch.setattr(train_dataset, "sample_positions", direct_sampling_is_forbidden)

    summary = trainer.train(max_steps=1)

    assert summary.steps == 1


@pytest.mark.parametrize("failure", ["gradient", "parameter"])
def test_non_finite_optimizer_state_writes_atomic_diagnostic_checkpoint(
    tmp_path: Path,
    failure: str,
) -> None:
    config = _config(tmp_path)
    train_dataset, _ = _datasets(tmp_path)
    model = TinyLLM(config.model)
    optimizer: torch.optim.Optimizer | None = None
    if failure == "gradient":
        model.token_embedding.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("nan"))
        )
    else:

        class CorruptingOptimizer(torch.optim.SGD):
            def step(self, closure=None):
                result = super().step(closure)
                with torch.no_grad():
                    self.param_groups[0]["params"][0].fill_(float("inf"))
                return result

        optimizer = CorruptingOptimizer(model.parameters(), lr=0.01)

    with pytest.raises(FloatingPointError, match=f"non-finite {failure}"):
        Trainer(config, model, train_dataset, optimizer=optimizer).train(max_steps=1)

    _assert_diagnostic_checkpoint(config, f"non-finite {failure}")


@pytest.mark.cuda
def test_non_finite_cuda_gradient_writes_diagnostic_checkpoint(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = _config(tmp_path, device="cuda")
    train_dataset, _ = _datasets(tmp_path)
    model = TinyLLM(config.model)
    model.token_embedding.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        Trainer(config, model, train_dataset).train(max_steps=1)

    _assert_diagnostic_checkpoint(config, "non-finite gradient")


def _assert_diagnostic_checkpoint(config: TinyLLMConfig, reason: str) -> None:
    checkpoint = config.training.checkpoint_dir / "diagnostic.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert payload["status"] == "diagnostic"
    assert reason in payload["reason"]
    assert isinstance(payload["model_state"], dict)
    assert isinstance(payload["optimizer_state"], dict)
    assert not list(checkpoint.parent.glob(f".{checkpoint.name}.*.tmp"))
