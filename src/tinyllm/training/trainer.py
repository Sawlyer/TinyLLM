from __future__ import annotations

import copy
import math
import random
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from tinyllm.config.schema import TinyLLMConfig
from tinyllm.data.loader import DeterministicBatchPrefetcher, TokenDataset
from tinyllm.data.validation import validate_dataset_for_model
from tinyllm.model.transformer import TinyLLM
from tinyllm.training import checkpoint as checkpointing
from tinyllm.training.metrics import MetricLogger
from tinyllm.training.optim import build_optimizer, cosine_lr
from tinyllm.training.precision import PrecisionPolicy


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    steps: int
    initial_loss: float
    final_loss: float
    tokens_processed: int
    elapsed_seconds: float
    validation_loss: float | None = None


class Trainer:
    """Single-device TinyLLM trainer with explicit numeric safety policies."""

    def __init__(
        self,
        config: TinyLLMConfig,
        model: TinyLLM,
        train_dataset: TokenDataset,
        validation_dataset: TokenDataset | None = None,
        *,
        optimizer: Optimizer | None = None,
        metric_logger: MetricLogger | None = None,
        validation_batches: int = 8,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(config, TinyLLMConfig):
            raise TypeError("config must be a TinyLLMConfig")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(train_dataset, TokenDataset):
            raise TypeError("train_dataset must be a TokenDataset")
        if validation_dataset is not None and not isinstance(validation_dataset, TokenDataset):
            raise TypeError("validation_dataset must be a TokenDataset or None")
        if type(validation_batches) is not int or validation_batches <= 0:
            raise ValueError("validation_batches must be a positive integer")

        self.config = config
        self.device = _resolve_device(config.training.device)
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        validate_dataset_for_model(train_dataset, config.model)
        if validation_dataset is not None:
            validate_dataset_for_model(validation_dataset, config.model)
        self.batch_pipeline = DeterministicBatchPrefetcher(
            train_dataset,
            config.training.micro_batch_size,
            prefetch_batches=2,
            pin_memory=True,
        )
        self.validation_batches = validation_batches
        self._clock = time.perf_counter if clock is None else clock
        self.precision = PrecisionPolicy.create(config.training.precision, self.device)
        self.model = self.precision.prepare_model(self.model)
        self.optimizer = optimizer or build_optimizer(
            self.model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.metric_logger = metric_logger
        train_artifact = train_dataset.manifest.artifacts[train_dataset.split]
        self.artifact_identity = checkpointing.ArtifactIdentity(
            model_config=config.model.model_dump(mode="json"),
            tokenizer_sha256=train_dataset.manifest.tokenizer_sha256,
            corpus_sha256=train_artifact.sha256,
        )
        self._step = 0
        self._tokens_processed = 0
        self._interrupt_requested = False
        self._training_model: nn.Module = self.model
        if config.training.compile or self.precision.requires_compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("configured training precision requires torch.compile support")
            self._training_model = torch.compile(self.model)

    def train(
        self, max_steps: int | None = None, *, overwrite_checkpoints: bool = False
    ) -> TrainingSummary:
        total_steps = self.config.training.max_steps
        steps = total_steps if max_steps is None else max_steps
        if type(steps) is not int or steps <= 0:
            raise ValueError("max_steps must be a positive integer or None")
        if steps > total_steps:
            raise ValueError("max_steps cannot exceed config.training.max_steps")
        if steps <= self._step:
            raise ValueError("max_steps must be greater than the current training step")

        logger = self.metric_logger or MetricLogger(self.config.logging.run_dir)
        wall_start = self._clock()
        interval_training_seconds = 0.0
        training_segment_start: float | None = None
        interval_tokens = 0
        tokens_processed = self._tokens_processed
        last_validation_loss: float | None = None
        self._interrupt_requested = False
        try:
            with self._sigint_checkpoint_request():
                initial_loss = self._monitor_loss(self.train_dataset)
                self._require_finite_scalar(initial_loss, "loss before training")
                self._raise_after_interrupt_checkpoint()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                for step in range(self._step + 1, steps + 1):
                    self._raise_after_interrupt_checkpoint()
                    if training_segment_start is None:
                        self._synchronize_for_metrics()
                        training_segment_start = self._clock()
                    self._training_model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated_loss: Tensor | None = None
                    for _ in range(self.config.training.gradient_accumulation_steps):
                        inputs, targets = self._sample_training_batch()
                        with self.precision.autocast():
                            output = self._training_model(inputs, targets)
                            loss = output.loss
                        if loss is None:
                            raise RuntimeError("model did not return a training loss")
                        self._require_finite_tensor(loss, f"loss at step {step}")
                        detached_loss = loss.detach()
                        accumulated_loss = (
                            detached_loss
                            if accumulated_loss is None
                            else accumulated_loss + detached_loss
                        )
                        normalized_loss = loss / self.config.training.gradient_accumulation_steps
                        self.precision.backward(normalized_loss)

                    if accumulated_loss is None:
                        raise RuntimeError("gradient accumulation produced no loss")
                    self.precision.unscale_(self.optimizer)
                    self._require_finite_gradients(step)
                    gradient_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.max_grad_norm
                    )
                    self._require_finite_tensor(gradient_norm, f"gradient norm at step {step}")
                    learning_rate = cosine_lr(
                        step,
                        self.config.training.warmup_steps,
                        total_steps,
                        self.config.training.learning_rate,
                        self.config.training.min_learning_rate,
                    )
                    for group in self.optimizer.param_groups:
                        group["lr"] = learning_rate
                    self.precision.step(self.optimizer)
                    self._require_finite_parameters(step)

                    step_tokens = (
                        self.config.training.micro_batch_size
                        * self.train_dataset.sequence_length
                        * self.config.training.gradient_accumulation_steps
                    )
                    tokens_processed += step_tokens
                    interval_tokens += step_tokens
                    self._step = step
                    self._tokens_processed = tokens_processed
                    self._raise_after_interrupt_checkpoint()

                    mean_loss_tensor = (
                        accumulated_loss / self.config.training.gradient_accumulation_steps
                    )
                    should_validate = self.validation_dataset is not None and (
                        step % self.config.logging.validation_interval == 0 or step == steps
                    )
                    should_log = step % self.config.logging.log_interval == 0 or step == steps
                    if should_validate or should_log:
                        self._synchronize_for_metrics()
                        training_segment_end = self._clock()
                        interval_training_seconds += training_segment_end - training_segment_start
                        training_segment_start = None
                    if should_validate:
                        last_validation_loss = self._validation_loss(self.validation_dataset)
                        self._raise_after_interrupt_checkpoint()

                    if should_log:
                        mean_loss = float(mean_loss_tensor)
                        metric_time = self._clock()
                        metrics: dict[str, int | float] = {
                            "tokens_processed": tokens_processed,
                            "learning_rate": learning_rate,
                            "loss": mean_loss,
                            "perplexity": _safe_perplexity(mean_loss),
                            "tokens_per_second": interval_tokens
                            / max(interval_training_seconds, 1e-12),
                            "elapsed_seconds": metric_time - wall_start,
                            "peak_vram_bytes": self._peak_vram_bytes(),
                            "gradient_norm": float(gradient_norm),
                        }
                        if last_validation_loss is not None:
                            metrics["validation_loss"] = last_validation_loss
                            metrics["validation_perplexity"] = _safe_perplexity(
                                last_validation_loss
                            )
                        logger.log(step, metrics)
                        self._raise_after_interrupt_checkpoint()
                        interval_tokens = 0
                        interval_training_seconds = 0.0

                    if step % self.config.logging.checkpoint_interval == 0:
                        checkpoint_path = self.config.training.checkpoint_dir / (
                            f"step-{step:08d}.pt"
                        )
                        if checkpoint_path.exists() and not overwrite_checkpoints:
                            raise FileExistsError(
                                f"checkpoint already exists: {checkpoint_path}; "
                                "pass --overwrite to replace it"
                            )
                        self.save_checkpoint(checkpoint_path)

                final_loss = self._monitor_loss(self.train_dataset)
                self._require_finite_scalar(final_loss, "loss after training")
                self._raise_after_interrupt_checkpoint()
                if self.validation_dataset is not None and last_validation_loss is None:
                    last_validation_loss = self._validation_loss(self.validation_dataset)
                return TrainingSummary(
                    steps=steps,
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    tokens_processed=tokens_processed,
                    elapsed_seconds=self._clock() - wall_start,
                    validation_loss=last_validation_loss,
                )
        except FloatingPointError as error:
            diagnostic_path = self.config.training.checkpoint_dir / "diagnostic.pt"
            self.save_checkpoint(
                diagnostic_path,
                status="diagnostic",
                reason=str(error),
            )
            raise FloatingPointError(
                f"{error}; diagnostic checkpoint: {diagnostic_path}"
            ) from error
        finally:
            logger.close()

    def save_checkpoint(
        self,
        path: Path,
        *,
        status: str = "resumable",
        reason: str | None = None,
    ) -> None:
        """Capture all mutable training state at the current step boundary."""
        cuda_rng_metadata = self._cuda_rng_metadata()
        cuda_rng_states = torch.cuda.get_rng_state_all() if cuda_rng_metadata["available"] else []
        state = checkpointing.TrainingState(
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state={
                "last_step": self._step,
                "total_steps": self.config.training.max_steps,
            },
            precision_state=self.precision.state_dict(),
            step=self._step,
            tokens_processed=self._tokens_processed,
            config_snapshot=self.config.model_dump(mode="json"),
            identity=self.artifact_identity,
            python_rng_state=random.getstate(),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=cuda_rng_states,
            cuda_rng_metadata=cuda_rng_metadata,
            loader_rng_state=self.train_dataset.state_dict(),
            status=status,
            reason=reason,
        )
        checkpointing.save_checkpoint(path, state)

    def load_checkpoint(self, path: Path) -> None:
        """Restore state before any subsequent training batch is sampled."""
        state = checkpointing.load_checkpoint(path, self.artifact_identity)
        self._validate_checkpoint_state(state)
        previous_state = self._capture_mutable_state()
        try:
            self._apply_checkpoint_state(state)
        except Exception:
            try:
                self._restore_mutable_state(previous_state)
            except Exception as rollback_error:
                raise RuntimeError("checkpoint load failed and rollback was unsuccessful") from (
                    rollback_error
                )
            raise

    def _validate_checkpoint_state(self, state: checkpointing.TrainingState) -> None:
        self._validate_config_snapshot(state.config_snapshot)
        scheduler_state = dict(state.scheduler_state)
        if set(scheduler_state) != {"last_step", "total_steps"}:
            raise ValueError("checkpoint scheduler state has invalid fields")
        if scheduler_state.get("last_step") != state.step:
            raise ValueError("checkpoint scheduler step is inconsistent")
        if scheduler_state.get("total_steps") != self.config.training.max_steps:
            raise ValueError("checkpoint scheduler configuration is incompatible")
        if state.step > self.config.training.max_steps:
            raise ValueError("checkpoint step exceeds configured total steps")
        tokens_per_step = (
            self.config.training.micro_batch_size
            * self.train_dataset.sequence_length
            * self.config.training.gradient_accumulation_steps
        )
        if state.tokens_processed != state.step * tokens_per_step:
            raise ValueError("checkpoint token count is inconsistent with step")
        self.train_dataset.validate_state_dict(state.loader_rng_state)
        self._validate_python_rng_state(state.python_rng_state)
        self._validate_torch_rng_state(state.torch_rng_state, "PyTorch RNG state")
        self._validate_cuda_rng_state(state)

    def _apply_checkpoint_state(self, state: checkpointing.TrainingState) -> None:
        self.model.load_state_dict(state.model_state)
        self.optimizer.load_state_dict(state.optimizer_state)
        self.precision.load_state_dict(dict(state.precision_state))
        self.train_dataset.load_state_dict(state.loader_rng_state)
        self.batch_pipeline.reset()
        random.setstate(state.python_rng_state)
        torch.set_rng_state(state.torch_rng_state)
        if state.cuda_rng_states:
            torch.cuda.set_rng_state_all(state.cuda_rng_states)
        self._step = state.step
        self._tokens_processed = state.tokens_processed

    def _validate_config_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        model_snapshot = snapshot.get("model")
        training_snapshot = snapshot.get("training")
        if not isinstance(model_snapshot, Mapping) or not isinstance(training_snapshot, Mapping):
            raise ValueError("checkpoint configuration snapshot is invalid")
        current_model = self.config.model.model_dump(mode="json")
        if dict(model_snapshot) != current_model:
            raise ValueError("checkpoint model configuration is incompatible")
        fields = (
            "learning_rate",
            "min_learning_rate",
            "warmup_steps",
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "precision",
            "weight_decay",
            "max_grad_norm",
            "compile",
        )
        current_training = self.config.training.model_dump(mode="json")
        for field in fields:
            if training_snapshot.get(field) != current_training[field]:
                raise ValueError(f"checkpoint training {field} is incompatible")

    def _validate_cuda_rng_state(self, state: checkpointing.TrainingState) -> None:
        metadata = dict(state.cuda_rng_metadata)
        required = {
            "available",
            "device_count",
            "active_device",
            "training_device_type",
            "training_device_index",
        }
        if set(metadata) != required:
            raise ValueError("checkpoint CUDA RNG metadata has invalid fields")
        available = metadata["available"]
        device_count = metadata["device_count"]
        active_device = metadata["active_device"]
        if type(available) is not bool or type(device_count) is not int:
            raise ValueError("checkpoint CUDA RNG metadata is invalid")
        if available:
            if device_count <= 0 or type(active_device) is not int:
                raise ValueError("checkpoint CUDA RNG metadata is inconsistent")
            if not 0 <= active_device < device_count:
                raise ValueError("checkpoint CUDA active device is invalid")
            if len(state.cuda_rng_states) != device_count:
                raise ValueError("checkpoint CUDA RNG state count is inconsistent")
        elif device_count != 0 or active_device is not None or state.cuda_rng_states:
            raise ValueError("checkpoint CUDA RNG metadata is inconsistent")
        for device_index, cuda_state in enumerate(state.cuda_rng_states):
            if (
                not isinstance(cuda_state, Tensor)
                or cuda_state.device.type != "cpu"
                or cuda_state.dtype != torch.uint8
                or cuda_state.numel() == 0
            ):
                raise ValueError("checkpoint CUDA RNG state is invalid")
            try:
                torch.Generator(device=f"cuda:{device_index}").set_state(cuda_state)
            except RuntimeError as error:
                raise ValueError("checkpoint CUDA RNG state is invalid") from error
        current = self._cuda_rng_metadata()
        for field in required:
            if metadata[field] != current[field]:
                raise ValueError(f"checkpoint CUDA/device metadata {field} is incompatible")

    def _cuda_rng_metadata(self) -> dict[str, bool | int | str | None]:
        available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if available else 0
        active_device = torch.cuda.current_device() if available else None
        training_device_index = self.device.index
        if self.device.type == "cuda" and training_device_index is None:
            training_device_index = active_device
        return {
            "available": available,
            "device_count": device_count,
            "active_device": active_device,
            "training_device_type": self.device.type,
            "training_device_index": training_device_index,
        }

    @staticmethod
    def _validate_python_rng_state(state: tuple[Any, ...]) -> None:
        try:
            random.Random().setstate(state)
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint Python RNG state is invalid") from error

    @staticmethod
    def _validate_torch_rng_state(state: Tensor, label: str) -> None:
        if not isinstance(state, Tensor) or state.device.type != "cpu":
            raise ValueError(f"checkpoint {label} must be a CPU tensor")
        probe = torch.Generator(device="cpu")
        try:
            probe.set_state(state)
        except RuntimeError as error:
            raise ValueError(f"checkpoint {label} is invalid") from error

    def _capture_mutable_state(self) -> dict[str, Any]:
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        return {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "precision": copy.deepcopy(self.precision.state_dict()),
            "loader": copy.deepcopy(self.train_dataset.state_dict()),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state().clone(),
            "cuda_rng": [cuda_state.clone() for cuda_state in cuda_states],
            "step": self._step,
            "tokens_processed": self._tokens_processed,
        }

    def _restore_mutable_state(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.precision.load_state_dict(state["precision"])
        self.train_dataset.load_state_dict(state["loader"])
        self.batch_pipeline.reset()
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self._step = state["step"]
        self._tokens_processed = state["tokens_processed"]

    @contextmanager
    def _sigint_checkpoint_request(self) -> Iterator[None]:
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous_handler = signal.getsignal(signal.SIGINT)

        def request_checkpoint(_signum: int, _frame: Any) -> None:
            self._interrupt_requested = True

        signal.signal(signal.SIGINT, request_checkpoint)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous_handler)

    def _raise_after_interrupt_checkpoint(self) -> None:
        if not self._interrupt_requested:
            return
        self.save_checkpoint(self.config.training.checkpoint_dir / "interrupt.pt")
        raise KeyboardInterrupt

    def _sample_training_batch(self) -> tuple[Tensor, Tensor]:
        inputs, targets = self.batch_pipeline.next_batch()
        non_blocking = self.device.type == "cuda"
        return (
            inputs.to(self.device, non_blocking=non_blocking),
            targets.to(self.device, non_blocking=non_blocking),
        )

    def _monitor_loss(self, dataset: TokenDataset) -> float:
        count = min(self.config.training.micro_batch_size, len(dataset))
        positions = torch.arange(count, dtype=torch.long)
        return self._loss_for_positions(dataset, positions)

    def _validation_loss(self, dataset: TokenDataset | None) -> float:
        if dataset is None:
            raise ValueError("validation dataset is unavailable")
        total_loss = 0.0
        total_examples = 0
        batch_size = self.config.training.micro_batch_size
        limit = min(len(dataset), batch_size * self.validation_batches)
        for start in range(0, limit, batch_size):
            positions = torch.arange(start, min(start + batch_size, limit), dtype=torch.long)
            batch_loss = self._loss_for_positions(dataset, positions)
            self._require_finite_scalar(batch_loss, "validation loss")
            total_loss += batch_loss * positions.numel()
            total_examples += positions.numel()
        if total_examples == 0:
            raise ValueError("validation dataset must contain at least one window")
        return total_loss / total_examples

    def _loss_for_positions(self, dataset: TokenDataset, positions: Tensor) -> float:
        was_training = self._training_model.training
        self._training_model.eval()
        try:
            inputs, targets = dataset.batch(positions)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            with torch.no_grad(), self.precision.autocast():
                output = self._training_model(inputs, targets)
            if output.loss is None:
                raise RuntimeError("model did not return an evaluation loss")
            return float(output.loss.detach())
        finally:
            self._training_model.train(was_training)

    def _require_finite_gradients(self, step: int) -> None:
        gradients = [
            parameter.grad for parameter in self.model.parameters() if parameter.grad is not None
        ]
        self._require_finite_group(gradients, f"gradients at step {step}")

    def _require_finite_parameters(self, step: int) -> None:
        self._require_finite_group(list(self.model.parameters()), f"parameters at step {step}")

    @staticmethod
    def _require_finite_tensor(value: Tensor, label: str) -> None:
        try:
            torch._assert_async(torch.isfinite(value).all(), f"non-finite {label}")
        except RuntimeError as error:
            raise FloatingPointError(f"non-finite {label}") from error

    @staticmethod
    def _require_finite_group(values: list[Tensor], label: str) -> None:
        if not values:
            return
        infinity_norms = torch._foreach_norm(values, float("inf"))
        all_finite = torch.stack(infinity_norms).isfinite().all()
        try:
            torch._assert_async(all_finite, f"non-finite {label}")
        except RuntimeError as error:
            raise FloatingPointError(f"non-finite {label}") from error

    @staticmethod
    def _require_finite_scalar(value: float, label: str) -> None:
        if not math.isfinite(value):
            raise FloatingPointError(f"non-finite {label}")

    def _synchronize_for_metrics(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _peak_vram_bytes(self) -> int:
        if self.device.type != "cuda":
            return 0
        return torch.cuda.max_memory_allocated(self.device)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid training device: {name!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training requested but CUDA is unavailable; use training.device=cpu"
        )
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS training requested but MPS is unavailable")
    return device


def _safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 80.0))
