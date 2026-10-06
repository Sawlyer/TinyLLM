"""Atomic, identity-checked training checkpoint serialization."""

from __future__ import annotations

import os
import pickle
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tinyllm.config.schema import ModelConfig
from tinyllm.errors import CheckpointCompatibilityError, CheckpointFormatError
from tinyllm.model.transformer import TinyLLM

LEGACY_CHECKPOINT_VERSION = 2
CHECKPOINT_VERSION = 3


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Model and input artifacts that must match before training can resume."""

    model_config: Mapping[str, Any]
    tokenizer_sha256: str
    corpus_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_config, Mapping) or not self.model_config:
            raise ValueError("model_config must be a non-empty mapping")
        if not isinstance(self.tokenizer_sha256, str) or not self.tokenizer_sha256:
            raise ValueError("tokenizer_sha256 must be a non-empty string")
        if not isinstance(self.corpus_sha256, str) or not self.corpus_sha256:
            raise ValueError("corpus_sha256 must be a non-empty string")


@dataclass(slots=True)
class TrainingState:
    """Complete state required to continue at the next batch boundary."""

    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any]
    scheduler_state: Mapping[str, Any]
    precision_state: Mapping[str, Any]
    step: int
    tokens_processed: int
    config_snapshot: Mapping[str, Any]
    identity: ArtifactIdentity
    python_rng_state: tuple[Any, ...]
    torch_rng_state: Tensor
    cuda_rng_states: list[Tensor]
    cuda_rng_metadata: Mapping[str, Any]
    loader_rng_state: Mapping[str, Any]
    status: str = "resumable"
    reason: str | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("model_state", self.model_state),
            ("optimizer_state", self.optimizer_state),
            ("scheduler_state", self.scheduler_state),
            ("precision_state", self.precision_state),
            ("config_snapshot", self.config_snapshot),
            ("cuda_rng_metadata", self.cuda_rng_metadata),
            ("loader_rng_state", self.loader_rng_state),
        ):
            if not isinstance(value, Mapping):
                raise TypeError(f"{field} must be a mapping")
        if type(self.step) is not int or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if type(self.tokens_processed) is not int or self.tokens_processed < 0:
            raise ValueError("tokens_processed must be a non-negative integer")
        if not isinstance(self.identity, ArtifactIdentity):
            raise TypeError("identity must be an ArtifactIdentity")
        if not isinstance(self.python_rng_state, tuple):
            raise TypeError("python_rng_state must be a tuple")
        if not isinstance(self.torch_rng_state, Tensor):
            raise TypeError("torch_rng_state must be a torch.Tensor")
        if not isinstance(self.cuda_rng_states, list) or not all(
            isinstance(state, Tensor) for state in self.cuda_rng_states
        ):
            raise TypeError("cuda_rng_states must be a list of torch.Tensor values")
        if self.status not in {"resumable", "diagnostic"}:
            raise ValueError("checkpoint status must be resumable or diagnostic")
        if self.status == "diagnostic":
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("diagnostic checkpoint requires a reason")
        elif self.reason is not None:
            raise ValueError("resumable checkpoint cannot contain a diagnostic reason")


@dataclass(frozen=True, slots=True)
class LoadedModelCheckpoint:
    """Inference-only checkpoint view without optimizer or RNG restoration."""

    model: nn.Module
    config_snapshot: Mapping[str, Any]
    identity: ArtifactIdentity
    step: int
    tokens_processed: int


def save_checkpoint(path: Path, state: TrainingState) -> None:
    """Durably replace ``path`` with one complete checkpoint."""
    if not isinstance(state, TrainingState):
        raise TypeError("state must be a TrainingState")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as checkpoint_file:
            torch.save(_state_to_payload(state), checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, expected: ArtifactIdentity) -> TrainingState:
    """Load a tensor-only checkpoint after validating artifact compatibility."""
    if not isinstance(expected, ArtifactIdentity):
        raise TypeError("expected must be an ArtifactIdentity")
    payload = _load_payload(Path(path))
    state = _payload_to_state(payload)
    if state.status != "resumable":
        raise CheckpointCompatibilityError(
            f"diagnostic checkpoint cannot resume training: {state.reason}"
        )
    _require_compatible_identity(state.identity, expected)
    return state


def load_model_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    model_factory: Callable[[ModelConfig], nn.Module] = TinyLLM,
) -> LoadedModelCheckpoint:
    """Strictly load only model-facing fields, ignoring training topology."""
    payload = _load_payload(Path(path))
    version = payload.get("version")
    if version not in {LEGACY_CHECKPOINT_VERSION, CHECKPOINT_VERSION}:
        raise CheckpointFormatError(f"unsupported checkpoint version: {version!r}")
    required = {"model_state", "identity"}
    if not required.issubset(payload):
        raise CheckpointFormatError("invalid checkpoint model fields")
    status = payload.get("status", "resumable")
    if status != "resumable":
        raise CheckpointCompatibilityError(
            f"diagnostic checkpoint cannot be used for inference: {payload.get('reason')}"
        )
    identity_payload = _require_mapping(payload["identity"], "identity")
    identity_fields = {"model_config", "tokenizer_sha256", "corpus_sha256"}
    if set(identity_payload) != identity_fields:
        raise CheckpointFormatError("invalid checkpoint identity fields")
    identity = ArtifactIdentity(
        model_config=_require_mapping(identity_payload["model_config"], "model_config"),
        tokenizer_sha256=identity_payload["tokenizer_sha256"],
        corpus_sha256=identity_payload["corpus_sha256"],
    )
    try:
        model_config = ModelConfig.model_validate(identity.model_config)
    except ValueError as error:
        raise CheckpointFormatError("checkpoint model configuration is invalid") from error
    model = model_factory(model_config)
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    model_state = _require_mapping(payload["model_state"], "model_state")
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise CheckpointFormatError("checkpoint model_state did not strictly load") from error
    model.eval()
    model.to(device=torch.device(device))
    config_snapshot = payload.get("config_snapshot", {})
    if not isinstance(config_snapshot, Mapping):
        raise CheckpointFormatError("checkpoint configuration snapshot must be a mapping")
    step = payload.get("step", 0)
    tokens_processed = payload.get("tokens_processed", 0)
    if type(step) is not int or step < 0:
        raise CheckpointFormatError("checkpoint step must be a non-negative integer")
    if type(tokens_processed) is not int or tokens_processed < 0:
        raise CheckpointFormatError("checkpoint token count must be a non-negative integer")
    return LoadedModelCheckpoint(
        model=model,
        config_snapshot=dict(config_snapshot),
        identity=identity,
        step=step,
        tokens_processed=tokens_processed,
    )


def _state_to_payload(state: TrainingState) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "model_state": dict(state.model_state),
        "optimizer_state": dict(state.optimizer_state),
        "scheduler_state": dict(state.scheduler_state),
        "precision_state": dict(state.precision_state),
        "step": state.step,
        "tokens_processed": state.tokens_processed,
        "config_snapshot": dict(state.config_snapshot),
        "identity": {
            "model_config": dict(state.identity.model_config),
            "tokenizer_sha256": state.identity.tokenizer_sha256,
            "corpus_sha256": state.identity.corpus_sha256,
        },
        "python_rng_state": state.python_rng_state,
        "torch_rng_state": state.torch_rng_state,
        "cuda_rng_states": state.cuda_rng_states,
        "cuda_rng_metadata": dict(state.cuda_rng_metadata),
        "loader_rng_state": dict(state.loader_rng_state),
        "status": state.status,
        "reason": state.reason,
    }


def _payload_to_state(payload: Mapping[str, Any]) -> TrainingState:
    base_required = {
        "version",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "precision_state",
        "step",
        "tokens_processed",
        "config_snapshot",
        "identity",
        "python_rng_state",
        "torch_rng_state",
        "cuda_rng_states",
        "cuda_rng_metadata",
        "loader_rng_state",
    }
    version = payload.get("version")
    if version == CHECKPOINT_VERSION:
        required = base_required | {"status", "reason"}
    elif version == LEGACY_CHECKPOINT_VERSION:
        required = base_required
    else:
        raise ValueError(f"unsupported checkpoint version: {version!r}")
    if set(payload) != required:
        raise ValueError("invalid checkpoint fields")
    identity_payload = _require_mapping(payload["identity"], "identity")
    identity_fields = {"model_config", "tokenizer_sha256", "corpus_sha256"}
    if set(identity_payload) != identity_fields:
        raise ValueError("invalid checkpoint identity fields")
    cuda_rng_states = payload["cuda_rng_states"]
    if not isinstance(cuda_rng_states, list):
        raise ValueError("cuda_rng_states must be a list")
    python_rng_state = payload["python_rng_state"]
    if not isinstance(python_rng_state, tuple):
        raise ValueError("python_rng_state must be a tuple")
    return TrainingState(
        model_state=_require_mapping(payload["model_state"], "model_state"),
        optimizer_state=_require_mapping(payload["optimizer_state"], "optimizer_state"),
        scheduler_state=_require_mapping(payload["scheduler_state"], "scheduler_state"),
        precision_state=_require_mapping(payload["precision_state"], "precision_state"),
        step=payload["step"],
        tokens_processed=payload["tokens_processed"],
        config_snapshot=_require_mapping(payload["config_snapshot"], "config_snapshot"),
        identity=ArtifactIdentity(
            model_config=_require_mapping(identity_payload["model_config"], "model_config"),
            tokenizer_sha256=identity_payload["tokenizer_sha256"],
            corpus_sha256=identity_payload["corpus_sha256"],
        ),
        python_rng_state=python_rng_state,
        torch_rng_state=payload["torch_rng_state"],
        cuda_rng_states=cuda_rng_states,
        cuda_rng_metadata=_require_mapping(payload["cuda_rng_metadata"], "cuda_rng_metadata"),
        loader_rng_state=_require_mapping(payload["loader_rng_state"], "loader_rng_state"),
        status=payload.get("status", "resumable"),
        reason=payload.get("reason"),
    )


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise CheckpointFormatError(f"invalid checkpoint: {path}") from error
    except RuntimeError as error:
        if not _is_serialization_runtime_error(error):
            raise
        raise CheckpointFormatError(f"invalid checkpoint: {path}") from error
    if not isinstance(payload, Mapping):
        raise CheckpointFormatError("invalid checkpoint root")
    return payload


def _is_serialization_runtime_error(error: RuntimeError) -> bool:
    message = str(error)
    return any(
        marker in message
        for marker in (
            "PytorchStreamReader",
            "Invalid magic number",
            "failed finding central directory",
        )
    )


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_compatible_identity(actual: ArtifactIdentity, expected: ArtifactIdentity) -> None:
    if dict(actual.model_config) != dict(expected.model_config):
        raise ValueError("checkpoint model identity is incompatible")
    if actual.tokenizer_sha256 != expected.tokenizer_sha256:
        raise ValueError("checkpoint tokenizer identity is incompatible")
    if actual.corpus_sha256 != expected.corpus_sha256:
        raise ValueError("checkpoint corpus identity is incompatible")
