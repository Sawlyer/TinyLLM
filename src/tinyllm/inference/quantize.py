"""Atomic, versioned, round-trip-validated inference checkpoint export."""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import pickle
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import torch
from torch import Tensor, nn

from tinyllm.config.schema import ModelConfig
from tinyllm.data.artifacts import sha256_file
from tinyllm.errors import (
    CheckpointCompatibilityError,
    CheckpointFormatError,
    OptionalBackendUnavailableError,
)
from tinyllm.model.transformer import TinyLLM
from tinyllm.training.checkpoint import load_model_checkpoint

LEGACY_QUANTIZED_FORMAT_VERSION = 2
QUANTIZED_FORMAT_VERSION = 3
SUPPORTED_RECIPES = ("bf16", "fp16", "int8-weight-only", "int4-weight-only")
_WEIGHT_ONLY_RECIPES = frozenset({"int8-weight-only", "int4-weight-only"})
_BASE_TENSOR_TYPES = frozenset({"torch.Tensor", "torch.nn.parameter.Parameter"})


@dataclass(frozen=True, slots=True)
class BackendRecipeMetadata:
    """Exact backend/config identity needed to trust a serialized tensor subclass."""

    torchao_version: str
    config_class: str
    config_parameters: Mapping[str, object]
    tensor_subclass_serialization_version: int


class QuantizationBackend(Protocol):
    def recipe_metadata(self, recipe: str) -> BackendRecipeMetadata: ...

    def quantize(
        self,
        model: nn.Module,
        recipe: str,
        device: torch.device,
    ) -> BackendRecipeMetadata: ...

    def prepare_for_load(
        self,
        model: nn.Module,
        recipe: str,
        manifest: QuantizedManifest,
    ) -> nn.Module: ...

    def is_quantized_linear(self, module: nn.Module) -> bool: ...


class _TorchAOQuantizationBackend:
    def __init__(
        self,
        quantize_in_place: Callable[[nn.Module, object], None],
        config_factories: Mapping[str, Callable[[], object]],
        torchao_version: str,
    ) -> None:
        self._quantize_in_place = quantize_in_place
        self._config_factories = config_factories
        self._torchao_version = torchao_version

    def recipe_metadata(self, recipe: str) -> BackendRecipeMetadata:
        return self._metadata_for_config(self._config_factories[recipe]())

    def quantize(
        self,
        model: nn.Module,
        recipe: str,
        device: torch.device,
    ) -> BackendRecipeMetadata:
        del device
        config = self._config_factories[recipe]()
        metadata = self._metadata_for_config(config)
        self._quantize_in_place(model, config)
        return metadata

    def prepare_for_load(
        self,
        model: nn.Module,
        recipe: str,
        manifest: QuantizedManifest,
    ) -> nn.Module:
        del recipe, manifest
        # TorchAO state_dict values carry tensor subclasses. Official reload uses
        # load_state_dict(assign=True) on an unquantized module skeleton.
        return model

    def is_quantized_linear(self, module: nn.Module) -> bool:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor):
            return False
        return _effective_tensor_type(weight).startswith("torchao.")

    def _metadata_for_config(self, config: object) -> BackendRecipeMetadata:
        return BackendRecipeMetadata(
            torchao_version=self._torchao_version,
            config_class=_qualified_type(config),
            config_parameters=_serialize_config_parameters(config),
            tensor_subclass_serialization_version=1,
        )


@dataclass(frozen=True, slots=True)
class QuantizedManifest:
    recipe: str
    source_checkpoint_sha256: str
    tokenizer_sha256: str | None
    torch_version: str
    torchao_version: str | None
    config_class: str
    config_parameters: Mapping[str, object]
    tensor_subclass_types: tuple[str, ...]
    tensor_subclass_serialization_version: int
    eligible_linear_count: int
    converted_linear_count: int
    partial_quantization: bool
    unconverted_linear_names: tuple[str, ...]
    converted_linear_types: Mapping[str, Mapping[str, str]]
    weight_tying_preserved: bool
    format_version: int = QUANTIZED_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version not in {
            LEGACY_QUANTIZED_FORMAT_VERSION,
            QUANTIZED_FORMAT_VERSION,
        }:
            raise ValueError("unsupported quantized manifest version")
        if self.recipe not in SUPPORTED_RECIPES:
            raise ValueError("unsupported quantization recipe")
        digest = self.source_checkpoint_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("source_checkpoint_sha256 must be a lowercase SHA-256 digest")
        if self.tokenizer_sha256 is not None:
            tokenizer_digest = self.tokenizer_sha256
            if len(tokenizer_digest) != 64 or any(
                character not in "0123456789abcdef" for character in tokenizer_digest
            ):
                raise ValueError("tokenizer_sha256 must be a lowercase SHA-256 digest")
        if self.format_version == QUANTIZED_FORMAT_VERSION and self.tokenizer_sha256 is None:
            raise ValueError("quantized format v3 requires tokenizer_sha256")
        if not self.torch_version:
            raise ValueError("torch_version must be non-empty")
        if not self.config_class or not isinstance(self.config_parameters, Mapping):
            raise ValueError("quantization configuration metadata is invalid")
        if self.tensor_subclass_serialization_version != 1:
            raise ValueError("unsupported tensor subclass serialization version")
        if self.eligible_linear_count < 0 or self.converted_linear_count < 0:
            raise ValueError("linear conversion counts must be non-negative")
        if self.converted_linear_count > self.eligible_linear_count:
            raise ValueError("converted linear count exceeds eligible count")
        if len(self.converted_linear_types) != self.converted_linear_count:
            raise ValueError("converted linear type count is inconsistent")
        expected_unconverted = self.eligible_linear_count - self.converted_linear_count
        if len(self.unconverted_linear_names) != expected_unconverted:
            raise ValueError("unconverted linear count is inconsistent")
        if self.partial_quantization != bool(self.unconverted_linear_names):
            raise ValueError("partial_quantization is inconsistent")
        if self.recipe in _WEIGHT_ONLY_RECIPES:
            if not self.torchao_version:
                raise ValueError("TorchAO recipe requires torchao_version")
            if self.converted_linear_count == 0:
                raise ValueError("TorchAO recipe requires converted Linear weights")
        else:
            expected_dtype = "torch.bfloat16" if self.recipe == "bf16" else "torch.float16"
            if (
                self.torchao_version is not None
                or self.config_class != "torch.dtype"
                or dict(self.config_parameters) != {"dtype": expected_dtype}
                or self.tensor_subclass_types
            ):
                raise ValueError("direct precision configuration metadata is incompatible")
            if self.eligible_linear_count or self.converted_linear_count:
                raise ValueError("direct precision recipe cannot declare quantized Linear modules")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "format_version": self.format_version,
            "recipe": self.recipe,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "torch_version": self.torch_version,
            "torchao_version": self.torchao_version,
            "config_class": self.config_class,
            "config_parameters": dict(self.config_parameters),
            "tensor_subclass_types": list(self.tensor_subclass_types),
            "tensor_subclass_serialization_version": (self.tensor_subclass_serialization_version),
            "eligible_linear_count": self.eligible_linear_count,
            "converted_linear_count": self.converted_linear_count,
            "partial_quantization": self.partial_quantization,
            "unconverted_linear_names": list(self.unconverted_linear_names),
            "converted_linear_types": {
                name: dict(types) for name, types in self.converted_linear_types.items()
            },
            "weight_tying_preserved": self.weight_tying_preserved,
        }
        if self.format_version == QUANTIZED_FORMAT_VERSION:
            payload["tokenizer_sha256"] = self.tokenizer_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QuantizedManifest:
        base_required = {
            "format_version",
            "recipe",
            "source_checkpoint_sha256",
            "torch_version",
            "torchao_version",
            "config_class",
            "config_parameters",
            "tensor_subclass_types",
            "tensor_subclass_serialization_version",
            "eligible_linear_count",
            "converted_linear_count",
            "partial_quantization",
            "unconverted_linear_names",
            "converted_linear_types",
            "weight_tying_preserved",
        }
        version = _require_int(payload.get("format_version"), "format_version")
        if version == QUANTIZED_FORMAT_VERSION:
            required = base_required | {"tokenizer_sha256"}
        elif version == LEGACY_QUANTIZED_FORMAT_VERSION:
            required = base_required
        else:
            raise ValueError(f"unsupported quantized format version: {version!r}")
        if set(payload) != required:
            raise ValueError("invalid quantized manifest fields")
        config_parameters = _require_mapping(payload["config_parameters"], "config_parameters")
        raw_types = _require_mapping(payload["converted_linear_types"], "converted_linear_types")
        converted_types: dict[str, dict[str, str]] = {}
        for name, type_payload in raw_types.items():
            if not isinstance(name, str):
                raise ValueError("converted Linear names must be strings")
            values = _require_mapping(type_payload, f"converted_linear_types.{name}")
            if set(values) != {"module", "weight"} or not all(
                isinstance(value, str) and value for value in values.values()
            ):
                raise ValueError("converted Linear type metadata is invalid")
            converted_types[name] = dict(values)
        return cls(
            format_version=version,
            recipe=_require_string(payload["recipe"], "recipe"),
            source_checkpoint_sha256=_require_string(
                payload["source_checkpoint_sha256"], "source_checkpoint_sha256"
            ),
            tokenizer_sha256=(
                None
                if version == LEGACY_QUANTIZED_FORMAT_VERSION
                else _require_string(payload["tokenizer_sha256"], "tokenizer_sha256")
            ),
            torch_version=_require_string(payload["torch_version"], "torch_version"),
            torchao_version=_optional_string(payload["torchao_version"], "torchao_version"),
            config_class=_require_string(payload["config_class"], "config_class"),
            config_parameters=dict(config_parameters),
            tensor_subclass_types=_string_tuple(
                payload["tensor_subclass_types"], "tensor_subclass_types"
            ),
            tensor_subclass_serialization_version=_require_int(
                payload["tensor_subclass_serialization_version"],
                "tensor_subclass_serialization_version",
            ),
            eligible_linear_count=_require_int(
                payload["eligible_linear_count"], "eligible_linear_count"
            ),
            converted_linear_count=_require_int(
                payload["converted_linear_count"], "converted_linear_count"
            ),
            partial_quantization=_require_bool(
                payload["partial_quantization"], "partial_quantization"
            ),
            unconverted_linear_names=_string_tuple(
                payload["unconverted_linear_names"], "unconverted_linear_names"
            ),
            converted_linear_types=converted_types,
            weight_tying_preserved=_require_bool(
                payload["weight_tying_preserved"], "weight_tying_preserved"
            ),
        )


@dataclass(frozen=True, slots=True)
class LoadedQuantizedCheckpoint:
    model: nn.Module
    manifest: QuantizedManifest
    model_config: ModelConfig


def _default_model_factory(config: ModelConfig) -> nn.Module:
    return TinyLLM(config)


@dataclass(slots=True)
class QuantizeRuntime:
    backend_loader: Callable[[], QuantizationBackend] | None = None
    model_factory: Callable[[ModelConfig], nn.Module] = _default_model_factory
    save_payload: Callable[[object, BinaryIO], None] = torch.save


def quantize_checkpoint(
    source: Path,
    output: Path,
    recipe: str,
    *,
    overwrite: bool = False,
    device: torch.device | str = "cpu",
    runtime: QuantizeRuntime | None = None,
) -> QuantizedManifest:
    """Export only checkpoints proven reloadable and numerically executable."""
    normalized_recipe = _validate_recipe(recipe)
    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("source and output paths must differ")
    if (output_path.exists() or _sidecar_path(output_path).exists()) and not overwrite:
        raise FileExistsError(f"quantized output already exists: {output_path}")

    active_runtime = QuantizeRuntime() if runtime is None else runtime
    resolved_device = torch.device(device)
    source_checkpoint = load_model_checkpoint(
        source_path,
        device=resolved_device,
        model_factory=active_runtime.model_factory,
    )
    model = source_checkpoint.model
    model_config = ModelConfig.model_validate(source_checkpoint.identity.model_config)
    tokenizer_sha256 = source_checkpoint.identity.tokenizer_sha256

    if normalized_recipe in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if normalized_recipe == "bf16" else torch.float16
        model.to(dtype=dtype)
        manifest = _direct_manifest(
            normalized_recipe,
            source_path,
            dtype,
            model,
            tokenizer_sha256,
        )
    else:
        model.to(dtype=torch.bfloat16)
        manifest = _quantize_model(
            model,
            model_config,
            normalized_recipe,
            source_path,
            resolved_device,
            active_runtime,
            tokenizer_sha256,
        )

    reference_input = _round_trip_input(model_config, resolved_device)
    reference_logits = _forward_logits(model, reference_input)
    export_payload = {
        "format_version": QUANTIZED_FORMAT_VERSION,
        "manifest": manifest.to_dict(),
        "model_config": model_config.model_dump(mode="json"),
        "model_state": dict(model.state_dict()),
    }

    def validate_round_trip(temporary: Path) -> None:
        loaded = load_quantized_checkpoint(
            temporary,
            device=resolved_device,
            runtime=active_runtime,
        )
        reloaded_logits = _forward_logits(loaded.model, reference_input)
        try:
            torch.testing.assert_close(reloaded_logits, reference_logits, rtol=1e-5, atol=1e-5)
        except AssertionError as error:
            raise RuntimeError("quantized checkpoint numeric round-trip failed") from error

    _atomic_save(
        output_path,
        export_payload,
        overwrite=overwrite,
        save_payload=active_runtime.save_payload,
        validator=validate_round_trip,
        manifest=manifest,
    )
    return manifest


def load_quantized_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
    runtime: QuantizeRuntime | None = None,
) -> LoadedQuantizedCheckpoint:
    """Reconstruct and strictly load a version-compatible inference export."""
    checkpoint_path = Path(path)
    active_runtime = QuantizeRuntime() if runtime is None else runtime
    preflight = _load_preflight(checkpoint_path)
    backend: QuantizationBackend | None = None
    safe_globals: list[object] = []
    if preflight is not None:
        preflight_manifest, expected_checkpoint_sha256 = preflight
        if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
            raise CheckpointFormatError(
                "quantized checkpoint checksum differs from preflight manifest"
            )
        backend = _validate_manifest_environment(
            preflight_manifest,
            active_runtime,
        )
        safe_globals = _safe_globals_for_checkpoint(checkpoint_path, preflight_manifest)
    else:
        unsafe_globals = _unsafe_globals_in_checkpoint(checkpoint_path)
        if unsafe_globals:
            raise CheckpointFormatError(
                "quantized checkpoint with tensor subclasses lacks trusted preflight metadata; "
                "re-export with current TinyLLM"
            )
    payload = _load_payload(
        checkpoint_path,
        "quantized checkpoint",
        safe_globals=safe_globals,
    )
    required = {"format_version", "manifest", "model_config", "model_state"}
    if set(payload) != required or payload["format_version"] not in {
        LEGACY_QUANTIZED_FORMAT_VERSION,
        QUANTIZED_FORMAT_VERSION,
    }:
        raise ValueError("invalid quantized checkpoint fields or format version")
    manifest = QuantizedManifest.from_dict(_require_mapping(payload["manifest"], "manifest"))
    if manifest.format_version != payload["format_version"]:
        raise CheckpointFormatError("quantized checkpoint format metadata is inconsistent")
    if preflight is not None and manifest != preflight[0]:
        raise CheckpointFormatError("embedded quantized manifest differs from preflight metadata")
    if preflight is None:
        backend = _validate_manifest_environment(manifest, active_runtime)
    model_config = _parse_model_config(payload["model_config"], "model_config")
    model_state = _require_mapping(payload["model_state"], "model_state")
    resolved_device = torch.device(device)
    model = _reconstruct_model(model_config, active_runtime)
    if manifest.recipe in {"bf16", "fp16"}:
        _validate_direct_precision_state(
            model,
            model_state,
            manifest.recipe,
            stage="before assign",
        )
    if manifest.recipe in _WEIGHT_ONLY_RECIPES:
        if backend is None:
            raise CheckpointCompatibilityError("TorchAO backend validation was not completed")
        if not manifest.weight_tying_preserved:
            _break_output_weight_tie(model)
        model = backend.prepare_for_load(model, manifest.recipe, manifest)
    _assign_exported_state(model, model_state)
    _reconcile_output_weight_tie(model, manifest.weight_tying_preserved)
    if manifest.recipe in {"bf16", "fp16"}:
        _validate_direct_precision_state(
            model,
            model.state_dict(),
            manifest.recipe,
            stage="after assign",
        )
    model.eval()
    model.to(device=resolved_device)
    if backend is not None:
        _validate_loaded_conversions(model, manifest, backend)
    return LoadedQuantizedCheckpoint(model=model, manifest=manifest, model_config=model_config)


def _validate_recipe(recipe: str) -> str:
    if not isinstance(recipe, str):
        raise ValueError("recipe must be a string")
    normalized = recipe.lower()
    if normalized not in SUPPORTED_RECIPES:
        choices = ", ".join(SUPPORTED_RECIPES)
        raise ValueError(f"recipe must be one of {choices}")
    return normalized


def _load_payload(
    path: Path,
    description: str,
    *,
    safe_globals: Sequence[object] = (),
) -> Mapping[str, Any]:
    try:
        if safe_globals:
            with torch.serialization.safe_globals(list(safe_globals)):
                payload = torch.load(path, map_location="cpu", weights_only=True)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise CheckpointFormatError(f"invalid {description}: {path}") from error
    except RuntimeError as error:
        if not _is_serialization_runtime_error(error):
            raise
        raise CheckpointFormatError(f"invalid {description}: {path}") from error
    if not isinstance(payload, Mapping):
        raise CheckpointFormatError(f"{description} root must be a mapping")
    return payload


def _load_preflight(path: Path) -> tuple[QuantizedManifest, str] | None:
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointFormatError(f"invalid quantized preflight metadata: {sidecar}") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "preflight_version",
        "checkpoint_sha256",
        "manifest",
    }:
        raise CheckpointFormatError("invalid quantized preflight fields")
    if payload["preflight_version"] != 1:
        raise CheckpointFormatError("unsupported quantized preflight version")
    checkpoint_sha256 = _require_string(
        payload["checkpoint_sha256"], "checkpoint_sha256"
    )
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        raise CheckpointFormatError("invalid quantized preflight checkpoint checksum")
    try:
        manifest = QuantizedManifest.from_dict(
            _require_mapping(payload["manifest"], "manifest")
        )
    except (TypeError, ValueError) as error:
        raise CheckpointFormatError("invalid quantized preflight manifest") from error
    if manifest.format_version != QUANTIZED_FORMAT_VERSION:
        raise CheckpointFormatError("legacy quantized checkpoints cannot use v1 preflight")
    return manifest, checkpoint_sha256


def _validate_manifest_environment(
    manifest: QuantizedManifest,
    runtime: QuantizeRuntime,
) -> QuantizationBackend | None:
    current_torch_version = str(torch.__version__)
    if manifest.torch_version != current_torch_version:
        raise CheckpointCompatibilityError(
            "quantized checkpoint PyTorch version mismatch: "
            f"exported {manifest.torch_version}, current {current_torch_version}"
        )
    if manifest.recipe not in _WEIGHT_ONLY_RECIPES:
        return None
    backend = _resolve_backend(manifest.recipe, runtime)
    current_metadata = backend.recipe_metadata(manifest.recipe)
    _require_compatible_backend(manifest, current_metadata)
    return backend


def _safe_globals_for_checkpoint(
    path: Path,
    manifest: QuantizedManifest,
) -> list[object]:
    unsafe_globals = _unsafe_globals_in_checkpoint(path)
    if not unsafe_globals:
        return []
    if manifest.recipe not in _WEIGHT_ONLY_RECIPES:
        raise CheckpointFormatError("direct precision checkpoint contains unsafe globals")
    declared_types = set(manifest.tensor_subclass_types)
    unexpected = set(unsafe_globals) - declared_types
    if unexpected:
        raise CheckpointFormatError(
            "quantized checkpoint requests undeclared unsafe globals: "
            + ", ".join(sorted(unexpected))
        )
    return [_resolve_safe_global(name) for name in unsafe_globals]


def _unsafe_globals_in_checkpoint(path: Path) -> list[str]:
    try:
        return list(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    except (OSError, EOFError, ValueError, pickle.UnpicklingError) as error:
        raise CheckpointFormatError(f"invalid quantized checkpoint: {path}") from error
    except RuntimeError as error:
        if not _is_serialization_runtime_error(error):
            raise
        raise CheckpointFormatError(f"invalid quantized checkpoint: {path}") from error


def _resolve_safe_global(qualified_name: str) -> object:
    if not qualified_name.startswith("torchao."):
        raise CheckpointFormatError(f"unsafe global is outside TorchAO: {qualified_name}")
    parts = qualified_name.split(".")
    module: object | None = None
    attribute_parts: list[str] = []
    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise OptionalBackendUnavailableError(
                    f"TorchAO global dependency is unavailable: {error}"
                ) from error
            continue
        attribute_parts = parts[split_index:]
        break
    if module is None:
        raise CheckpointFormatError(f"cannot resolve declared TorchAO global: {qualified_name}")
    resolved = module
    try:
        for attribute in attribute_parts:
            resolved = getattr(resolved, attribute)
    except AttributeError as error:
        raise CheckpointFormatError(
            f"cannot resolve declared TorchAO global: {qualified_name}"
        ) from error
    resolved_name = f"{getattr(resolved, '__module__', '')}.{getattr(resolved, '__qualname__', '')}"
    if resolved_name != qualified_name:
        raise CheckpointFormatError(f"TorchAO global identity mismatch: {qualified_name}")
    return resolved


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


def _parse_model_config(value: object, field_name: str) -> ModelConfig:
    raw_config = _require_mapping(value, field_name)
    try:
        return ModelConfig.model_validate(raw_config)
    except ValueError as error:
        raise ValueError(f"{field_name} is invalid") from error


def _validate_direct_precision_state(
    model: nn.Module,
    model_state: Mapping[str, Any],
    recipe: str,
    *,
    stage: str,
) -> None:
    expected_dtype = torch.bfloat16 if recipe == "bf16" else torch.float16
    parameters = dict(model.named_parameters(remove_duplicate=False))
    mismatches: list[str] = []
    for name, value in model_state.items():
        parameter = parameters.get(name)
        parameter_requires_float = parameter is not None and (
            parameter.is_floating_point() or parameter.is_complex()
        )
        if not isinstance(value, Tensor):
            if parameter_requires_float:
                mismatches.append(f"{name}={type(value).__name__}")
            continue
        value_requires_recipe_dtype = value.is_floating_point() or value.is_complex()
        if (
            parameter_requires_float or value_requires_recipe_dtype
        ) and value.dtype != expected_dtype:
            mismatches.append(f"{name}={value.dtype}")
    if mismatches:
        details = ", ".join(mismatches)
        raise RuntimeError(
            f"direct precision dtype mismatch {stage}: {details}; expected {expected_dtype}"
        )


def _direct_manifest(
    recipe: str,
    source_path: Path,
    dtype: torch.dtype,
    model: nn.Module,
    tokenizer_sha256: str,
) -> QuantizedManifest:
    return QuantizedManifest(
        recipe=recipe,
        source_checkpoint_sha256=sha256_file(source_path),
        tokenizer_sha256=tokenizer_sha256,
        torch_version=str(torch.__version__),
        torchao_version=None,
        config_class="torch.dtype",
        config_parameters={"dtype": str(dtype)},
        tensor_subclass_types=(),
        tensor_subclass_serialization_version=1,
        eligible_linear_count=0,
        converted_linear_count=0,
        partial_quantization=False,
        unconverted_linear_names=(),
        converted_linear_types={},
        weight_tying_preserved=_output_weights_are_tied(model),
    )


def _quantize_model(
    model: nn.Module,
    model_config: ModelConfig,
    recipe: str,
    source_path: Path,
    device: torch.device,
    runtime: QuantizeRuntime,
    tokenizer_sha256: str,
) -> QuantizedManifest:
    del model_config
    backend = _resolve_backend(recipe, runtime)
    eligible_names = tuple(
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    )
    try:
        metadata = backend.quantize(model, recipe, device)
    except (ImportError, NotImplementedError, OptionalBackendUnavailableError) as error:
        raise OptionalBackendUnavailableError(
            f"{recipe} TorchAO kernel is unavailable for device {device}: {error}"
        ) from error
    converted_types, tensor_types = _inspect_converted_linears(
        model,
        eligible_names,
        backend,
    )
    converted_names = tuple(converted_types)
    if not converted_names:
        raise RuntimeError(f"{recipe} converted zero eligible Linear modules; export refused")
    unconverted_names = tuple(name for name in eligible_names if name not in converted_types)
    return QuantizedManifest(
        recipe=recipe,
        source_checkpoint_sha256=sha256_file(source_path),
        tokenizer_sha256=tokenizer_sha256,
        torch_version=str(torch.__version__),
        torchao_version=metadata.torchao_version,
        config_class=metadata.config_class,
        config_parameters=dict(metadata.config_parameters),
        tensor_subclass_types=tensor_types,
        tensor_subclass_serialization_version=(metadata.tensor_subclass_serialization_version),
        eligible_linear_count=len(eligible_names),
        converted_linear_count=len(converted_names),
        partial_quantization=bool(unconverted_names),
        unconverted_linear_names=unconverted_names,
        converted_linear_types=converted_types,
        weight_tying_preserved=_output_weights_are_tied(model),
    )


def _resolve_backend(recipe: str, runtime: QuantizeRuntime) -> QuantizationBackend:
    loader = runtime.backend_loader or _load_torchao_backend
    try:
        return loader()
    except (ImportError, AttributeError, OptionalBackendUnavailableError) as error:
        raise OptionalBackendUnavailableError(
            f"{recipe} requires optional TorchAO support: {error}"
        ) from error


def _inspect_converted_linears(
    model: nn.Module,
    eligible_names: Sequence[str],
    backend: QuantizationBackend,
) -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
    converted: dict[str, dict[str, str]] = {}
    tensor_types: set[str] = set()
    for name in eligible_names:
        module = model.get_submodule(name)
        if not backend.is_quantized_linear(module):
            continue
        weight_type = _effective_tensor_type(module.weight)
        converted[name] = {
            "module": _qualified_type(module),
            "weight": weight_type,
        }
        if weight_type not in _BASE_TENSOR_TYPES:
            tensor_types.add(weight_type)
    return converted, tuple(sorted(tensor_types))


def _require_compatible_backend(
    manifest: QuantizedManifest,
    current: BackendRecipeMetadata,
) -> None:
    if manifest.torchao_version != current.torchao_version:
        raise CheckpointCompatibilityError(
            "quantized checkpoint TorchAO version mismatch: "
            f"exported {manifest.torchao_version}, current {current.torchao_version}"
        )
    expected = (
        manifest.config_class,
        dict(manifest.config_parameters),
        manifest.tensor_subclass_serialization_version,
    )
    actual = (
        current.config_class,
        dict(current.config_parameters),
        current.tensor_subclass_serialization_version,
    )
    if actual != expected:
        raise CheckpointCompatibilityError(
            "quantized checkpoint quantization configuration is incompatible"
        )


def _reconstruct_model(model_config: ModelConfig, runtime: QuantizeRuntime) -> nn.Module:
    try:
        with torch.device("meta"):
            model = runtime.model_factory(model_config)
        if any(buffer.device.type == "meta" for buffer in model.buffers()):
            model = runtime.model_factory(model_config)
    except NotImplementedError:
        model = runtime.model_factory(model_config)
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module")
    return model


def _assign_exported_state(model: nn.Module, model_state: Mapping[str, Any]) -> None:
    try:
        model.load_state_dict(model_state, strict=True, assign=True)
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(
            "quantized model_state did not strictly load with assign=True"
        ) from error


def _validate_loaded_conversions(
    model: nn.Module,
    manifest: QuantizedManifest,
    backend: QuantizationBackend,
) -> None:
    expected_names = tuple(manifest.converted_linear_types)
    actual_types, tensor_types = _inspect_converted_linears(model, expected_names, backend)
    if actual_types != dict(manifest.converted_linear_types):
        raise RuntimeError("reloaded quantized Linear types differ from manifest")
    if tensor_types != manifest.tensor_subclass_types:
        raise RuntimeError("reloaded tensor subclass types differ from manifest")
    for name in manifest.unconverted_linear_names:
        if backend.is_quantized_linear(model.get_submodule(name)):
            raise RuntimeError("reloaded partial quantization differs from manifest")


def _round_trip_input(model_config: ModelConfig, device: torch.device) -> Tensor:
    sequence_length = min(3, model_config.max_seq_len)
    return torch.arange(sequence_length, device=device).remainder(model_config.vocab_size)[None, :]


def _forward_logits(model: nn.Module, input_ids: Tensor) -> Tensor:
    with torch.inference_mode():
        output = model(input_ids)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor) or not bool(torch.isfinite(logits).all()):
        raise RuntimeError("quantized checkpoint forward produced non-finite logits")
    return logits.detach().float().cpu()


def _output_weights_are_tied(model: nn.Module) -> bool:
    return bool(
        hasattr(model, "lm_head")
        and hasattr(model, "token_embedding")
        and model.lm_head.weight is model.token_embedding.weight
    )


def _break_output_weight_tie(model: nn.Module) -> None:
    if not _output_weights_are_tied(model):
        return
    weight = model.lm_head.weight
    model.lm_head.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)


def _reconcile_output_weight_tie(model: nn.Module, should_be_tied: bool) -> None:
    if should_be_tied:
        model.lm_head.weight = model.token_embedding.weight
    elif _output_weights_are_tied(model):
        _break_output_weight_tie(model)


def _effective_tensor_type(tensor: Tensor) -> str:
    direct_type = _qualified_type(tensor)
    if direct_type not in _BASE_TENSOR_TYPES:
        return direct_type
    data_type = _qualified_type(tensor.data)
    return data_type if data_type not in _BASE_TENSOR_TYPES else direct_type


def _load_torchao_backend() -> QuantizationBackend:
    try:
        torchao = importlib.import_module("torchao")
        quantization = importlib.import_module("torchao.quantization")
        torchao_version = getattr(torchao, "__version__")
        quantize_in_place = getattr(quantization, "quantize_")
        int8_config = getattr(quantization, "Int8WeightOnlyConfig")
        int4_config = getattr(quantization, "Int4WeightOnlyConfig")
        if not isinstance(torchao_version, str) or not torchao_version:
            raise AttributeError("torchao.__version__")
    except (ImportError, AttributeError) as error:
        raise OptionalBackendUnavailableError(
            "reloadable TorchAO Int8WeightOnlyConfig/Int4WeightOnlyConfig APIs are unavailable; "
            "install a compatible tinyllm[fp8] environment"
        ) from error
    factories = {
        "int8-weight-only": int8_config,
        "int4-weight-only": lambda: int4_config(group_size=32),
    }
    return _TorchAOQuantizationBackend(quantize_in_place, factories, torchao_version)


def _serialize_config_parameters(config: object) -> dict[str, object]:
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return {
            field.name: _json_metadata_value(getattr(config, field.name))
            for field in dataclasses.fields(config)
        }
    try:
        values = vars(config)
    except TypeError as error:
        raise RuntimeError("TorchAO config parameters are not reliably serializable") from error
    public = {
        name: _json_metadata_value(value)
        for name, value in values.items()
        if not name.startswith("_")
    }
    if not public:
        raise RuntimeError("TorchAO config exposes no serializable parameters")
    return public


def _json_metadata_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Enum):
        return {
            "enum_class": _qualified_type(value),
            "value": _json_metadata_value(value.value),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_metadata_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_metadata_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_metadata_value(item) for item in value]
    raise RuntimeError(
        f"TorchAO config value {_qualified_type(value)} is not reliably serializable"
    )


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _require_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name)


def _require_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(value)


def _atomic_save(
    destination: Path,
    payload: object,
    *,
    overwrite: bool,
    save_payload: Callable[[object, BinaryIO], None],
    validator: Callable[[Path], None],
    manifest: QuantizedManifest,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    destination_sidecar = _sidecar_path(destination)
    temporary_sidecar = _sidecar_path(temporary)
    try:
        with temporary.open("xb") as output_file:
            save_payload(payload, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        _write_preflight(temporary_sidecar, temporary, manifest)
        validator(temporary)
        if overwrite:
            _replace_export_pair(
                temporary,
                temporary_sidecar,
                destination,
                destination_sidecar,
            )
        else:
            os.link(temporary, destination)
            try:
                os.link(temporary_sidecar, destination_sidecar)
            except BaseException:
                destination.unlink(missing_ok=True)
                raise
    finally:
        temporary.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)


def _sidecar_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.name}.manifest.json")


def _write_preflight(
    path: Path,
    checkpoint: Path,
    manifest: QuantizedManifest,
) -> None:
    payload = {
        "preflight_version": 1,
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": manifest.to_dict(),
    }
    with path.open("x", encoding="utf-8", newline="\n") as output_file:
        json.dump(payload, output_file, sort_keys=True, separators=(",", ":"))
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())


def _replace_export_pair(
    temporary: Path,
    temporary_sidecar: Path,
    destination: Path,
    destination_sidecar: Path,
) -> None:
    identifier = uuid.uuid4().hex
    checkpoint_backup = destination.with_name(f".{destination.name}.{identifier}.bak")
    sidecar_backup = destination_sidecar.with_name(
        f".{destination_sidecar.name}.{identifier}.bak"
    )
    checkpoint_backed_up = False
    sidecar_backed_up = False
    try:
        if destination.exists():
            os.replace(destination, checkpoint_backup)
            checkpoint_backed_up = True
        if destination_sidecar.exists():
            os.replace(destination_sidecar, sidecar_backup)
            sidecar_backed_up = True
        os.replace(temporary, destination)
        os.replace(temporary_sidecar, destination_sidecar)
    except BaseException:
        destination.unlink(missing_ok=True)
        destination_sidecar.unlink(missing_ok=True)
        if checkpoint_backed_up:
            os.replace(checkpoint_backup, destination)
            checkpoint_backed_up = False
        if sidecar_backed_up:
            os.replace(sidecar_backup, destination_sidecar)
            sidecar_backed_up = False
        raise
    finally:
        checkpoint_backup.unlink(missing_ok=True)
        sidecar_backup.unlink(missing_ok=True)
