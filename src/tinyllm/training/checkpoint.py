"""Atomic, identity-checked training checkpoint serialization."""

from __future__ import annotations

import os
import pickle
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tinyllm.config.schema import ModelConfig
from tinyllm.errors import CheckpointCompatibilityError, CheckpointFormatError
from tinyllm.model.transformer import TinyLLM

LEGACY_CHECKPOINT_VERSION = 2
PREVIOUS_CHECKPOINT_VERSION = 3
CHECKPOINT_VERSION = PREVIOUS_CHECKPOINT_VERSION
CHECKPOINT_FORMAT_VERSION = 1
TRAINING_CHECKPOINT_TYPE = "tinyllm.training_checkpoint"
INFERENCE_WEIGHTS_TYPE = "tinyllm.inference_weights"


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
    artifact_type: str = TRAINING_CHECKPOINT_TYPE
    format_version: int = CHECKPOINT_FORMAT_VERSION
    created_at_utc: str | None = None
    producer_version: str | None = None


@dataclass(frozen=True, slots=True)
class CheckpointInspection:
    """Validated checkpoint metadata without constructing a model."""

    path: Path
    size_bytes: int
    step: int
    tokens_processed: int
    status: str
    reason: str | None
    model_config: Mapping[str, Any]
    tokenizer_sha256: str
    corpus_sha256: str
    artifact_type: str = TRAINING_CHECKPOINT_TYPE
    format_version: int = CHECKPOINT_FORMAT_VERSION
    created_at_utc: str | None = None
    producer_version: str | None = None
    migrated_from_version: int | None = None


def save_checkpoint(path: Path, state: TrainingState) -> None:
    """Durably replace ``path`` with one complete checkpoint."""
    if not isinstance(state, TrainingState):
        raise TypeError("state must be a TrainingState")
    _atomic_save_payload(Path(path), _state_to_payload(state))


def save_model_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    config_snapshot: Mapping[str, Any],
    identity: ArtifactIdentity,
    step: int,
    tokens_processed: int,
    overwrite: bool = False,
) -> None:
    """Write inference-only weights without optimizer, scheduler, or RNG state."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(config_snapshot, Mapping) or not config_snapshot:
        raise ValueError("config_snapshot must be a non-empty mapping")
    if not isinstance(identity, ArtifactIdentity):
        raise TypeError("identity must be an ArtifactIdentity")
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    if type(tokens_processed) is not int or tokens_processed < 0:
        raise ValueError("tokens_processed must be a non-negative integer")
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"inference weights already exist: {destination}")
    payload = {
        **_format_metadata(INFERENCE_WEIGHTS_TYPE),
        "model_state": {
            name: value.detach().cpu() if isinstance(value, Tensor) else value
            for name, value in model.state_dict().items()
        },
        "identity": _identity_to_payload(identity),
        "config_snapshot": dict(config_snapshot),
        "step": step,
        "tokens_processed": tokens_processed,
    }
    _atomic_save_payload(destination, payload)


def load_checkpoint(path: Path, expected: ArtifactIdentity) -> TrainingState:
    """Load a tensor-only checkpoint after validating artifact compatibility."""
    if not isinstance(expected, ArtifactIdentity):
        raise TypeError("expected must be an ArtifactIdentity")
    payload, _migrated_from = _normalize_checkpoint_payload(_load_payload(Path(path)))
    _require_artifact_type(payload, TRAINING_CHECKPOINT_TYPE)
    state = _payload_to_state(payload)
    if state.status != "resumable":
        raise CheckpointCompatibilityError(
            f"diagnostic checkpoint cannot resume training: {state.reason}"
        )
    _require_compatible_identity(state.identity, expected)
    return state


def inspect_checkpoint(path: Path) -> CheckpointInspection:
    """Validate checkpoint structure and return metadata without model allocation."""
    checkpoint_path = Path(path)
    payload, migrated_from = _normalize_checkpoint_payload(_load_payload(checkpoint_path))
    try:
        artifact_type = _require_supported_artifact_type(payload)
        if artifact_type == TRAINING_CHECKPOINT_TYPE:
            state = _payload_to_state(payload)
            identity = state.identity
            step = state.step
            tokens_processed = state.tokens_processed
            status = state.status
            reason = state.reason
        else:
            model_state, identity, step, tokens_processed, _snapshot = _inference_payload_fields(
                payload
            )
            if not model_state:
                raise ValueError("model_state must not be empty")
            status = "inference"
            reason = None
    except (TypeError, ValueError) as error:
        raise CheckpointFormatError(f"invalid checkpoint fields: {checkpoint_path}") from error
    try:
        size_bytes = checkpoint_path.stat().st_size
    except OSError as error:
        raise CheckpointFormatError(
            f"checkpoint disappeared during inspection: {checkpoint_path}"
        ) from error
    return CheckpointInspection(
        path=checkpoint_path,
        size_bytes=size_bytes,
        step=step,
        tokens_processed=tokens_processed,
        status=status,
        reason=reason,
        model_config=dict(identity.model_config),
        tokenizer_sha256=identity.tokenizer_sha256,
        corpus_sha256=identity.corpus_sha256,
        artifact_type=artifact_type,
        format_version=payload["format_version"],
        created_at_utc=payload["created_at_utc"],
        producer_version=payload["producer_version"],
        migrated_from_version=migrated_from,
    )


def load_model_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    model_factory: Callable[[ModelConfig], nn.Module] = TinyLLM,
) -> LoadedModelCheckpoint:
    """Strictly load only model-facing fields, ignoring training topology."""
    payload, _migrated_from = _normalize_checkpoint_payload(_load_payload(Path(path)))
    artifact_type = _require_supported_artifact_type(payload)
    if artifact_type == TRAINING_CHECKPOINT_TYPE:
        status = payload.get("status", "resumable")
        if status != "resumable":
            raise CheckpointCompatibilityError(
                f"diagnostic checkpoint cannot be used for inference: {payload.get('reason')}"
            )
        model_fields = _training_model_fields(payload)
    else:
        model_fields = _inference_payload_fields(payload)
    model_state, identity, step, tokens_processed, config_snapshot = model_fields
    try:
        model_config = ModelConfig.model_validate(identity.model_config)
    except ValueError as error:
        raise CheckpointFormatError("checkpoint model configuration is invalid") from error
    model = model_factory(model_config)
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise CheckpointFormatError("checkpoint model_state did not strictly load") from error
    model.eval()
    model.to(device=torch.device(device))
    return LoadedModelCheckpoint(
        model=model,
        config_snapshot=dict(config_snapshot),
        identity=identity,
        step=step,
        tokens_processed=tokens_processed,
        artifact_type=artifact_type,
        format_version=payload["format_version"],
        created_at_utc=payload["created_at_utc"],
        producer_version=payload["producer_version"],
    )


def _state_to_payload(state: TrainingState) -> dict[str, Any]:
    return {
        **_format_metadata(TRAINING_CHECKPOINT_TYPE),
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
    _require_artifact_type(payload, TRAINING_CHECKPOINT_TYPE)
    required = base_required | _FORMAT_METADATA_FIELDS | {"status", "reason"}
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
        status=payload["status"],
        reason=payload["reason"],
    )


_FORMAT_METADATA_FIELDS = {
    "artifact_type",
    "format_version",
    "created_at_utc",
    "producer_version",
}


def _format_metadata(artifact_type: str) -> dict[str, Any]:
    try:
        producer_version = metadata.version("tinyllm")
    except metadata.PackageNotFoundError:
        producer_version = "0+unknown"
    return {
        "artifact_type": artifact_type,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at_utc": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "producer_version": producer_version,
    }


def _normalize_checkpoint_payload(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int | None]:
    normalized = dict(payload)
    if "format_version" not in normalized:
        legacy_version = normalized.get("version")
        if type(legacy_version) is not int:
            raise CheckpointFormatError("checkpoint lacks a recognized format version")
        if legacy_version > PREVIOUS_CHECKPOINT_VERSION:
            raise CheckpointCompatibilityError(
                f"checkpoint version {legacy_version} is newer than this TinyLLM; "
                "upgrade TinyLLM to read it"
            )
        if legacy_version not in {LEGACY_CHECKPOINT_VERSION, PREVIOUS_CHECKPOINT_VERSION}:
            raise CheckpointCompatibilityError(
                f"legacy checkpoint version {legacy_version} is unsupported; "
                "convert it with a TinyLLM release that supports this format"
            )
        normalized.pop("version")
        normalized.update(
            {
                "artifact_type": TRAINING_CHECKPOINT_TYPE,
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "created_at_utc": None,
                "producer_version": None,
            }
        )
        if legacy_version == LEGACY_CHECKPOINT_VERSION:
            normalized.setdefault("status", "resumable")
            normalized.setdefault("reason", None)
        return normalized, legacy_version

    format_version = normalized["format_version"]
    if type(format_version) is not int:
        raise CheckpointFormatError("checkpoint format_version must be an integer")
    if format_version > CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint format_version {format_version} is newer than supported version "
            f"{CHECKPOINT_FORMAT_VERSION}; upgrade TinyLLM to read it"
        )
    if format_version < CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint format_version {format_version} is no longer supported; "
            "convert it with a TinyLLM release that supports this format"
        )
    missing_metadata = _FORMAT_METADATA_FIELDS - normalized.keys()
    if missing_metadata:
        fields = ", ".join(sorted(missing_metadata))
        raise CheckpointFormatError(f"checkpoint is missing format metadata: {fields}")
    artifact_type = normalized.get("artifact_type")
    if not isinstance(artifact_type, str) or artifact_type not in {
        TRAINING_CHECKPOINT_TYPE,
        INFERENCE_WEIGHTS_TYPE,
    }:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint artifact_type {artifact_type!r}; "
            "supported types are training checkpoints and inference weights"
        )
    created_at = normalized.get("created_at_utc")
    if created_at is not None:
        if not isinstance(created_at, str):
            raise CheckpointFormatError("checkpoint created_at_utc must be an ISO 8601 string")
        try:
            parsed_timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise CheckpointFormatError("checkpoint created_at_utc is invalid") from error
        if parsed_timestamp.utcoffset() != UTC.utcoffset(parsed_timestamp):
            raise CheckpointFormatError("checkpoint created_at_utc must use UTC")
    producer_version = normalized.get("producer_version")
    if producer_version is not None and (
        not isinstance(producer_version, str) or not producer_version
    ):
        raise CheckpointFormatError("checkpoint producer_version must be a non-empty string")
    return normalized, None


def _require_supported_artifact_type(payload: Mapping[str, Any]) -> str:
    artifact_type = payload["artifact_type"]
    if not isinstance(artifact_type, str) or artifact_type not in {
        TRAINING_CHECKPOINT_TYPE,
        INFERENCE_WEIGHTS_TYPE,
    }:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint artifact_type {artifact_type!r}; "
            "use a TinyLLM release that supports this artifact"
        )
    return artifact_type


def _require_artifact_type(payload: Mapping[str, Any], expected: str) -> None:
    actual = _require_supported_artifact_type(payload)
    if actual != expected:
        if actual == INFERENCE_WEIGHTS_TYPE:
            raise CheckpointCompatibilityError(
                "inference-only weights cannot resume training; provide a resumable "
                "training checkpoint"
            )
        raise CheckpointCompatibilityError(
            "training checkpoint cannot be loaded as inference weights"
        )


def _training_model_fields(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ArtifactIdentity, int, int, Mapping[str, Any]]:
    model_state = _require_mapping(payload.get("model_state"), "model_state")
    identity = _identity_from_payload(payload.get("identity"))
    config_snapshot = payload.get("config_snapshot", {})
    if not isinstance(config_snapshot, Mapping):
        raise CheckpointFormatError("checkpoint configuration snapshot must be a mapping")
    step = payload.get("step", 0)
    tokens_processed = payload.get("tokens_processed", 0)
    _validate_model_counters(step, tokens_processed)
    return model_state, identity, step, tokens_processed, config_snapshot


def _inference_payload_fields(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ArtifactIdentity, int, int, Mapping[str, Any]]:
    required = _FORMAT_METADATA_FIELDS | {
        "model_state",
        "identity",
        "config_snapshot",
        "step",
        "tokens_processed",
    }
    if set(payload) != required:
        raise ValueError("invalid inference weights fields")
    model_state = _require_mapping(payload["model_state"], "model_state")
    if not model_state:
        raise ValueError("model_state must not be empty")
    identity = _identity_from_payload(payload["identity"])
    config_snapshot = _require_mapping(payload["config_snapshot"], "config_snapshot")
    step = payload["step"]
    tokens_processed = payload["tokens_processed"]
    _validate_model_counters(step, tokens_processed)
    return model_state, identity, step, tokens_processed, config_snapshot


def _identity_from_payload(value: Any) -> ArtifactIdentity:
    identity_payload = _require_mapping(value, "identity")
    identity_fields = {"model_config", "tokenizer_sha256", "corpus_sha256"}
    if set(identity_payload) != identity_fields:
        raise CheckpointFormatError("invalid checkpoint identity fields")
    return ArtifactIdentity(
        model_config=_require_mapping(identity_payload["model_config"], "model_config"),
        tokenizer_sha256=identity_payload["tokenizer_sha256"],
        corpus_sha256=identity_payload["corpus_sha256"],
    )


def _identity_to_payload(identity: ArtifactIdentity) -> dict[str, Any]:
    return {
        "model_config": dict(identity.model_config),
        "tokenizer_sha256": identity.tokenizer_sha256,
        "corpus_sha256": identity.corpus_sha256,
    }


def _validate_model_counters(step: Any, tokens_processed: Any) -> None:
    if type(step) is not int or step < 0:
        raise CheckpointFormatError("checkpoint step must be a non-negative integer")
    if type(tokens_processed) is not int or tokens_processed < 0:
        raise CheckpointFormatError("checkpoint token count must be a non-negative integer")


def _atomic_save_payload(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as checkpoint_file:
            torch.save(dict(payload), checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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

