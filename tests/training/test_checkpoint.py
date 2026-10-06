import copy
import random
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tinyllm.config.load import load_config
from tinyllm.config.schema import TinyLLMConfig
from tinyllm.data.artifacts import MANIFEST_VERSION, CorpusManifest, write_token_artifact
from tinyllm.data.loader import TokenDataset
from tinyllm.model.transformer import TinyLLM
from tinyllm.training.checkpoint import (
    ArtifactIdentity,
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from tinyllm.training.trainer import Trainer

TOKENIZER_SHA256 = "a" * 64


class RecordingLogger:
    def __init__(self) -> None:
        self.steps: list[int] = []
        self.losses: list[float] = []

    def log(self, step: int, metrics: dict[str, int | float]) -> None:
        self.steps.append(step)
        self.losses.append(float(metrics["loss"]))

    def close(self) -> None:
        pass


def _config(
    tmp_path: Path,
    *,
    checkpoint_interval: int = 100,
    learning_rate: float = 0.03,
    validation_interval: int = 100,
) -> TinyLLMConfig:
    return load_config(
        Path("configs/tinyllm.yaml"),
        [
            "tokenizer.vocab_size=8",
            "model.vocab_size=8",
            "model.max_seq_len=4",
            "model.d_model=8",
            "model.n_layers=1",
            "model.n_heads=2",
            "model.n_kv_heads=1",
            "model.mlp_ratio=2.0",
            "model.dropout=0.2",
            "training.device=cpu",
            "training.precision=fp32",
            "training.micro_batch_size=2",
            "training.gradient_accumulation_steps=2",
            "training.max_steps=12",
            f"training.learning_rate={learning_rate}",
            "training.min_learning_rate=0.003",
            "training.warmup_steps=0",
            "training.weight_decay=0.01",
            f"training.checkpoint_dir={tmp_path / 'checkpoints'}",
            f"logging.run_dir={tmp_path / 'runs'}",
            "logging.log_interval=1",
            f"logging.validation_interval={validation_interval}",
            f"logging.checkpoint_interval={checkpoint_interval}",
        ],
    )


def _dataset(tmp_path: Path, *, seed: int = 7) -> TokenDataset:
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
        dataset_name="checkpoint-fixture",
        dataset_requested_revision=None,
        dataset_revision="injected-sha256:" + "b" * 64,
        dataset_fingerprint="checkpoint-fixture",
        text_field="text",
        seed=seed,
        validation_fraction=0.5,
        tokenizer_path=str(tmp_path / "tokenizer.json"),
        tokenizer_sha256=TOKENIZER_SHA256,
        tokenizer_config={"vocab_size": 8},
        data_config={
            "dataset_name": "checkpoint-fixture",
            "dataset_revision": None,
            "text_field": "text",
            "seed": seed,
            "validation_fraction": 0.5,
            "output_dir": str(output_dir),
        },
        document_counts={"train": 1, "validation": 1},
        artifacts=artifacts,
        tokenizer_vocab_size=8,
        special_token_ids={
            "<unk>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<pad>": 3,
        },
    )
    return TokenDataset(manifest, "train", 4, seed, artifact_dir=output_dir)


def _trainer(tmp_path: Path, logger: RecordingLogger) -> Trainer:
    config = _config(tmp_path)
    dataset = _dataset(tmp_path)
    random.seed(17)
    torch.manual_seed(23)
    return Trainer(config, TinyLLM(config.model), dataset, metric_logger=logger)


def _assert_state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def _checkpoint_state() -> TrainingState:
    generator = torch.Generator(device="cpu").manual_seed(31)
    return TrainingState(
        model_state={"weight": torch.arange(4, dtype=torch.float32)},
        optimizer_state={"state": {0: {"step": torch.tensor(3)}}, "param_groups": []},
        scheduler_state={"last_step": 3},
        precision_state={"scale": torch.tensor(1024.0)},
        step=3,
        tokens_processed=96,
        config_snapshot={"training": {"precision": "fp16"}},
        identity=ArtifactIdentity(
            model_config={"d_model": 8, "n_layers": 1},
            tokenizer_sha256=TOKENIZER_SHA256,
            corpus_sha256="b" * 64,
        ),
        python_rng_state=random.getstate(),
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_states=[],
        cuda_rng_metadata={
            "available": False,
            "device_count": 0,
            "active_device": None,
            "training_device_type": "cpu",
            "training_device_index": None,
        },
        loader_rng_state={"generator_state": generator.get_state()},
    )


def test_checkpoint_round_trip_preserves_complete_training_state(tmp_path: Path) -> None:
    state = _checkpoint_state()
    path = tmp_path / "nested" / "checkpoint.pt"

    save_checkpoint(path, state)
    loaded = load_checkpoint(path, state.identity)

    assert loaded.step == 3
    assert loaded.tokens_processed == 96
    assert loaded.scheduler_state == {"last_step": 3}
    assert loaded.config_snapshot == {"training": {"precision": "fp16"}}
    assert loaded.python_rng_state == state.python_rng_state
    assert torch.equal(loaded.model_state["weight"], state.model_state["weight"])
    assert torch.equal(loaded.precision_state["scale"], state.precision_state["scale"])
    assert torch.equal(loaded.torch_rng_state, state.torch_rng_state)
    assert torch.equal(
        loaded.loader_rng_state["generator_state"],
        state.loader_rng_state["generator_state"],
    )


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("tokenizer_sha256", "wrong", "tokenizer"),
        ("corpus_sha256", "wrong", "corpus"),
        ("model_config", {"d_model": 16, "n_layers": 1}, "model"),
    ],
)
def test_checkpoint_rejects_incompatible_identity(
    tmp_path: Path, field: str, wrong_value: Any, message: str
) -> None:
    state = _checkpoint_state()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, state)
    expected = replace(state.identity, **{field: wrong_value})

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path, expected)


def test_failed_save_keeps_previous_checkpoint_and_removes_temporary_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    original = _checkpoint_state()
    save_checkpoint(path, original)
    invalid = replace(original, optimizer_state={"unpicklable": lambda: None}, step=4)

    with pytest.raises(AttributeError):
        save_checkpoint(path, invalid)

    assert load_checkpoint(path, original.identity).step == 3
    assert not list(tmp_path.glob(".checkpoint.pt.*.tmp"))


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    uninterrupted_logger = RecordingLogger()
    uninterrupted = _trainer(tmp_path / "uninterrupted", uninterrupted_logger)
    uninterrupted.train(max_steps=12)

    before_logger = RecordingLogger()
    before = _trainer(tmp_path / "resumed", before_logger)
    before.train(max_steps=5)
    checkpoint_path = tmp_path / "resume.pt"
    before.save_checkpoint(checkpoint_path)

    after_logger = RecordingLogger()
    resumed = _trainer(tmp_path / "resumed-after", after_logger)
    resumed.load_checkpoint(checkpoint_path)
    resumed.train(max_steps=12)

    _assert_state_dict_equal(uninterrupted.model.state_dict(), resumed.model.state_dict())
    assert uninterrupted_logger.losses == pytest.approx(
        before_logger.losses + after_logger.losses, rel=0.0, abs=0.0
    )


def test_checkpoint_interval_writes_resumable_completed_step(tmp_path: Path) -> None:
    logger = RecordingLogger()
    config = _config(tmp_path, checkpoint_interval=2)
    dataset = _dataset(tmp_path)
    torch.manual_seed(37)
    trainer = Trainer(config, TinyLLM(config.model), dataset, metric_logger=logger)

    trainer.train(max_steps=3)

    checkpoint_path = config.training.checkpoint_dir / "step-00000002.pt"
    loaded = load_checkpoint(checkpoint_path, trainer.artifact_identity)
    assert loaded.step == 2
    assert loaded.tokens_processed == 32


def test_load_restores_python_torch_and_loader_rng_before_sampling(tmp_path: Path) -> None:
    logger = RecordingLogger()
    trainer = _trainer(tmp_path, logger)
    checkpoint_path = tmp_path / "rng.pt"
    trainer.save_checkpoint(checkpoint_path)
    expected_python = random.random()
    expected_torch = torch.rand(4)
    expected_positions = trainer.train_dataset.sample_positions(4)
    random.seed(999)
    torch.manual_seed(999)
    trainer.train_dataset.generator.manual_seed(999)

    trainer.load_checkpoint(checkpoint_path)

    assert random.random() == expected_python
    assert torch.equal(torch.rand(4), expected_torch)
    assert torch.equal(trainer.train_dataset.sample_positions(4), expected_positions)


def test_resume_rejects_behavioral_config_change_before_mutation(tmp_path: Path) -> None:
    source = _trainer(tmp_path / "source", RecordingLogger())
    checkpoint_path = tmp_path / "checkpoint.pt"
    source.save_checkpoint(checkpoint_path)
    target_config = _config(tmp_path / "target", learning_rate=0.02)
    target_dataset = _dataset(tmp_path / "target")
    torch.manual_seed(43)
    target = Trainer(
        target_config,
        TinyLLM(target_config.model),
        target_dataset,
        metric_logger=RecordingLogger(),
    )
    model_before = {name: value.clone() for name, value in target.model.state_dict().items()}

    with pytest.raises(ValueError, match="learning_rate"):
        target.load_checkpoint(checkpoint_path)

    _assert_state_dict_equal(model_before, target.model.state_dict())


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("split", "validation"),
        ("sequence_length", 5),
        ("token_count", 511),
        ("position_count", 506),
        ("generator_device", "cuda"),
    ],
)
def test_loader_state_rejects_mismatched_sampling_domain(
    tmp_path: Path, field: str, wrong_value: Any
) -> None:
    dataset = _dataset(tmp_path)
    state = dataset.state_dict()
    state[field] = wrong_value

    with pytest.raises(ValueError, match=field):
        dataset.load_state_dict(state)


def test_failed_load_with_bad_generator_is_transactional(tmp_path: Path) -> None:
    source = _trainer(tmp_path / "source", RecordingLogger())
    source.train(max_steps=2)
    checkpoint_path = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint_path)
    state = load_checkpoint(checkpoint_path, source.artifact_identity)
    invalid_loader_state = dict(state.loader_rng_state)
    invalid_loader_state["generator_state"] = torch.tensor([1], dtype=torch.uint8)
    invalid_path = tmp_path / "invalid.pt"
    save_checkpoint(invalid_path, replace(state, loader_rng_state=invalid_loader_state))

    target = _trainer(tmp_path / "target", RecordingLogger())
    target.train(max_steps=1)
    model_before = {name: value.clone() for name, value in target.model.state_dict().items()}
    optimizer_before = copy.deepcopy(target.optimizer.state_dict())
    python_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()
    loader_rng_before = target.train_dataset.get_rng_state()
    step_before = target._step
    tokens_before = target._tokens_processed

    with pytest.raises(ValueError, match="generator_state"):
        target.load_checkpoint(invalid_path)

    _assert_state_dict_equal(model_before, target.model.state_dict())
    _assert_nested_equal(optimizer_before, target.optimizer.state_dict())
    assert random.getstate() == python_rng_before
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert torch.equal(target.train_dataset.get_rng_state(), loader_rng_before)
    assert target._step == step_before
    assert target._tokens_processed == tokens_before


def test_apply_failure_rolls_back_model_optimizer_and_rng(tmp_path: Path) -> None:
    source = _trainer(tmp_path / "source", RecordingLogger())
    source.train(max_steps=2)
    checkpoint_path = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint_path)
    state = load_checkpoint(checkpoint_path, source.artifact_identity)
    invalid_path = tmp_path / "invalid.pt"
    save_checkpoint(invalid_path, replace(state, precision_state={"unexpected": 1}))

    target = _trainer(tmp_path / "target", RecordingLogger())
    target.train(max_steps=1)
    model_before = {name: value.clone() for name, value in target.model.state_dict().items()}
    optimizer_before = copy.deepcopy(target.optimizer.state_dict())
    python_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()
    loader_rng_before = target.train_dataset.get_rng_state()
    step_before = target._step
    tokens_before = target._tokens_processed

    with pytest.raises(ValueError, match="precision"):
        target.load_checkpoint(invalid_path)

    _assert_state_dict_equal(model_before, target.model.state_dict())
    _assert_nested_equal(optimizer_before, target.optimizer.state_dict())
    assert random.getstate() == python_rng_before
    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert torch.equal(target.train_dataset.get_rng_state(), loader_rng_before)
    assert target._step == step_before
    assert target._tokens_processed == tokens_before


def test_resume_rejects_incoherent_step_and_token_count(tmp_path: Path) -> None:
    source = _trainer(tmp_path / "source", RecordingLogger())
    source.train(max_steps=2)
    checkpoint_path = tmp_path / "source.pt"
    source.save_checkpoint(checkpoint_path)
    state = load_checkpoint(checkpoint_path, source.artifact_identity)
    invalid_path = tmp_path / "invalid.pt"
    save_checkpoint(invalid_path, replace(state, tokens_processed=state.tokens_processed + 1))
    target = _trainer(tmp_path / "target", RecordingLogger())

    with pytest.raises(ValueError, match="token count"):
        target.load_checkpoint(invalid_path)


def test_cpu_checkpoint_is_rejected_for_cuda_training_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _trainer(tmp_path / "source", RecordingLogger())
    checkpoint_path = tmp_path / "cpu.pt"
    source.save_checkpoint(checkpoint_path)
    target = _trainer(tmp_path / "target", RecordingLogger())
    model_before = {name: value.clone() for name, value in target.model.state_dict().items()}
    target.device = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    with pytest.raises(ValueError, match="device|CUDA"):
        target.load_checkpoint(checkpoint_path)

    _assert_state_dict_equal(model_before, target.model.state_dict())


def _capture_sigint_handler(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[int, Any], Any]:
    installed_handler: dict[int, Any] = {}
    previous_handler = signal.getsignal(signal.SIGINT)

    def capture_handler(number: int, handler: Any) -> Any:
        installed_handler[number] = handler
        return previous_handler

    monkeypatch.setattr(signal, "signal", capture_handler)
    return installed_handler, previous_handler


def test_sigint_during_logging_stops_before_next_training_step_and_restores_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_handler, previous_handler = _capture_sigint_handler(monkeypatch)

    class InterruptingLogger(RecordingLogger):
        def log(self, step: int, metrics: dict[str, int | float]) -> None:
            super().log(step, metrics)
            installed_handler[signal.SIGINT](signal.SIGINT, None)

    config = _config(tmp_path)
    dataset = _dataset(tmp_path)
    torch.manual_seed(47)
    trainer = Trainer(
        config,
        TinyLLM(config.model),
        dataset,
        metric_logger=InterruptingLogger(),
    )

    with pytest.raises(KeyboardInterrupt):
        trainer.train(max_steps=3)

    loaded = load_checkpoint(
        config.training.checkpoint_dir / "interrupt.pt", trainer.artifact_identity
    )
    assert loaded.step == 1
    assert installed_handler[signal.SIGINT] is previous_handler


def test_sigint_during_validation_stops_before_next_training_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_handler, _ = _capture_sigint_handler(monkeypatch)

    class InterruptingValidationTrainer(Trainer):
        def _validation_loss(self, dataset: TokenDataset | None) -> float:
            installed_handler[signal.SIGINT](signal.SIGINT, None)
            return super()._validation_loss(dataset)

    config = _config(tmp_path, validation_interval=1)
    dataset = _dataset(tmp_path)
    torch.manual_seed(53)
    trainer = InterruptingValidationTrainer(
        config,
        TinyLLM(config.model),
        dataset,
        dataset,
        metric_logger=RecordingLogger(),
    )

    with pytest.raises(KeyboardInterrupt):
        trainer.train(max_steps=3)

    loaded = load_checkpoint(
        config.training.checkpoint_dir / "interrupt.pt", trainer.artifact_identity
    )
    assert loaded.step == 1


def test_sigint_saves_completed_step_at_safe_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed_handler: dict[int, Any] = {}
    previous_handler = signal.getsignal(signal.SIGINT)

    def capture_handler(number: int, handler: Any) -> Any:
        installed_handler[number] = handler
        return previous_handler

    monkeypatch.setattr(signal, "signal", capture_handler)
    logger = RecordingLogger()
    config = _config(tmp_path)
    dataset = _dataset(tmp_path)
    torch.manual_seed(41)
    model = TinyLLM(config.model)

    class InterruptingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            result = super().step(closure)
            installed_handler[signal.SIGINT](signal.SIGINT, None)
            return result

    trainer = Trainer(
        config,
        model,
        dataset,
        optimizer=InterruptingAdamW(model.parameters(), lr=config.training.learning_rate),
        metric_logger=logger,
    )

    with pytest.raises(KeyboardInterrupt):
        trainer.train(max_steps=12)

    checkpoint_path = config.training.checkpoint_dir / "interrupt.pt"
    loaded = load_checkpoint(checkpoint_path, trainer.artifact_identity)
    assert loaded.step == 1
    assert loaded.tokens_processed == 16
    _assert_state_dict_equal(model.state_dict(), loaded.model_state)


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right
