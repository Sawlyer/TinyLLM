from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from tinyllm.config.schema import ModelConfig
from tinyllm.errors import CheckpointFormatError, OptionalBackendUnavailableError
from tinyllm.inference import quantize as quantize_module
from tinyllm.inference.quantize import (
    BackendRecipeMetadata,
    QuantizedManifest,
    QuantizeRuntime,
    load_quantized_checkpoint,
    quantize_checkpoint,
)
from tinyllm.model.transformer import TinyLLM


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        rope_theta=10_000.0,
        rms_norm_eps=1.0e-5,
    )


def _checkpoint(tmp_path: Path) -> Path:
    config = _model_config()
    model = TinyLLM(config)
    source = tmp_path / "source.pt"
    torch.save(
        {
            "version": 2,
            "model_state": model.state_dict(),
            "identity": {
                "model_config": config.model_dump(mode="json"),
                "tokenizer_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
            },
        },
        source,
    )
    return source


class FakeQuantizedLinear(nn.Linear):
    @classmethod
    def from_linear(cls, source: nn.Linear) -> FakeQuantizedLinear:
        converted = cls(
            source.in_features,
            source.out_features,
            bias=source.bias is not None,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        converted.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        if source.bias is not None:
            converted.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False)
        return converted


class CorruptAfterAssignTinyLLM(TinyLLM):
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if assign:
            self.final_norm.weight = nn.Parameter(self.final_norm.weight.float())
        return result


class IntegerBufferTinyLLM(TinyLLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.register_buffer("quantization_marker", torch.tensor(17), persistent=True)


class HonestFakeQuantizationBackend:
    def __init__(self, *, convert_limit: int | None = None, bits_offset: int = 0) -> None:
        self.convert_limit = convert_limit
        self.bits_offset = bits_offset
        self.quantize_calls: list[tuple[str, torch.device]] = []
        self.prepare_calls: list[str] = []

    def recipe_metadata(self, recipe: str) -> BackendRecipeMetadata:
        bits = (8 if recipe == "int8-weight-only" else 4) + self.bits_offset
        return BackendRecipeMetadata(
            torchao_version="fake-ao-1.2.3",
            config_class="tests.inference.test_quantize.FakeWeightOnlyConfig",
            config_parameters={"bits": bits, "group_size": 32},
            tensor_subclass_serialization_version=1,
        )

    def quantize(
        self,
        model: nn.Module,
        recipe: str,
        device: torch.device,
    ) -> BackendRecipeMetadata:
        self.quantize_calls.append((recipe, device))
        candidates = [
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and not isinstance(module, FakeQuantizedLinear)
        ]
        selected = candidates[: self.convert_limit]
        for name in selected:
            _replace_module(model, name, FakeQuantizedLinear.from_linear(model.get_submodule(name)))
        return self.recipe_metadata(recipe)

    def prepare_for_load(
        self,
        model: nn.Module,
        recipe: str,
        manifest: QuantizedManifest,
    ) -> nn.Module:
        self.prepare_calls.append(recipe)
        for name in manifest.converted_linear_types:
            module = model.get_submodule(name)
            _replace_module(model, name, FakeQuantizedLinear.from_linear(module))
        return model

    def is_quantized_linear(self, module: nn.Module) -> bool:
        return isinstance(module, FakeQuantizedLinear)


def _replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, replacement)


def _runtime(backend: HonestFakeQuantizationBackend) -> QuantizeRuntime:
    return QuantizeRuntime(backend_loader=lambda: backend)


@pytest.mark.parametrize(
    ("recipe", "expected_dtype"),
    [("bf16", torch.bfloat16), ("fp16", torch.float16)],
)
def test_direct_precision_export_strictly_loads_then_casts_and_round_trips(
    tmp_path: Path,
    recipe: str,
    expected_dtype: torch.dtype,
) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / f"{recipe}.pt"

    manifest = quantize_checkpoint(source, output, recipe)
    loaded = load_quantized_checkpoint(output)

    assert manifest.recipe == recipe
    assert manifest.source_checkpoint_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest.tokenizer_sha256 == "a" * 64
    assert manifest.torch_version == torch.__version__
    assert manifest.torchao_version is None
    assert manifest.config_class == "torch.dtype"
    assert manifest.config_parameters == {"dtype": str(expected_dtype)}
    assert loaded.model.lm_head.weight is loaded.model.token_embedding.weight
    assert loaded.model.lm_head.weight.dtype is expected_dtype
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        logits = loaded.model(input_ids).logits
    assert torch.isfinite(logits).all()


def test_direct_precision_export_rejects_non_strict_source_state(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    payload["model_state"].pop("final_norm.weight")
    torch.save(payload, source)

    with pytest.raises(RuntimeError, match="strictly load"):
        quantize_checkpoint(source, tmp_path / "output.pt", "bf16")


@pytest.mark.parametrize("recipe", ["int8-weight-only"])
def test_optional_recipe_records_real_conversion_types_counts_and_round_trips(
    tmp_path: Path,
    recipe: str,
) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / f"{recipe}.pt"
    backend = HonestFakeQuantizationBackend()
    runtime = _runtime(backend)

    manifest = quantize_checkpoint(source, output, recipe, runtime=runtime)
    loaded = load_quantized_checkpoint(output, runtime=runtime)

    assert backend.quantize_calls == [(recipe, torch.device("cpu"))]
    assert backend.prepare_calls.count(recipe) >= 2
    assert manifest.torchao_version == "fake-ao-1.2.3"
    assert manifest.config_class.endswith("FakeWeightOnlyConfig")
    assert manifest.config_parameters["bits"] in {4, 8}
    assert manifest.tensor_subclass_serialization_version == 1
    assert manifest.eligible_linear_count == 8
    assert manifest.converted_linear_count == 8
    assert not manifest.partial_quantization
    assert manifest.unconverted_linear_names == ()
    assert set(manifest.converted_linear_types) == {
        "blocks.0.attention.q_proj",
        "blocks.0.attention.k_proj",
        "blocks.0.attention.v_proj",
        "blocks.0.attention.out_proj",
        "blocks.0.feed_forward.gate_proj",
        "blocks.0.feed_forward.up_proj",
        "blocks.0.feed_forward.down_proj",
        "lm_head",
    }
    assert all(
        types["module"].endswith("FakeQuantizedLinear")
        for types in manifest.converted_linear_types.values()
    )
    assert loaded.model.lm_head.weight is not loaded.model.token_embedding.weight
    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        logits = loaded.model(input_ids).logits
    assert torch.isfinite(logits).all()


def test_zero_actual_quantized_linears_refuses_export(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "int8.pt"
    backend = HonestFakeQuantizationBackend(convert_limit=0)

    with pytest.raises(RuntimeError, match="converted zero.*Linear"):
        quantize_checkpoint(source, output, "int8-weight-only", runtime=_runtime(backend))

    assert not output.exists()


def test_partial_quantization_is_declared_with_exact_stats(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "partial.pt"
    backend = HonestFakeQuantizationBackend(convert_limit=1)
    runtime = _runtime(backend)

    manifest = quantize_checkpoint(source, output, "int8-weight-only", runtime=runtime)
    loaded = load_quantized_checkpoint(output, runtime=runtime)

    assert manifest.eligible_linear_count == 8
    assert manifest.converted_linear_count == 1
    assert manifest.partial_quantization
    assert len(manifest.unconverted_linear_names) == 7
    assert len(manifest.converted_linear_types) == 1
    assert sum(backend.is_quantized_linear(module) for module in loaded.model.modules()) == 1


def test_loader_rejects_exact_torch_version_mismatch(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "bf16.pt"
    quantize_checkpoint(source, output, "bf16")
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["manifest"]["torch_version"] = "0.0.0"
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)

    with pytest.raises(RuntimeError, match="PyTorch version.*0.0.0"):
        load_quantized_checkpoint(incompatible)


def test_loader_rejects_direct_precision_configuration_mismatch(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "bf16.pt"
    quantize_checkpoint(source, output, "bf16")
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["manifest"]["config_parameters"] = {"dtype": "torch.float16"}
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)

    with pytest.raises(ValueError, match="direct precision configuration"):
        load_quantized_checkpoint(incompatible)


@pytest.mark.parametrize(
    ("recipe", "wrong_dtype"),
    [("bf16", torch.float32), ("fp16", torch.bfloat16)],
)
def test_loader_rejects_wrong_floating_parameter_dtype_before_assign(
    tmp_path: Path,
    recipe: str,
    wrong_dtype: torch.dtype,
) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / f"{recipe}.pt"
    quantize_checkpoint(source, output, recipe)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    payload["model_state"]["final_norm.weight"] = payload["model_state"]["final_norm.weight"].to(
        dtype=wrong_dtype
    )
    incompatible = tmp_path / "incompatible.pt"
    torch.save(payload, incompatible)

    with pytest.raises(RuntimeError, match="before assign.*final_norm.weight"):
        load_quantized_checkpoint(incompatible)


def test_loader_rejects_wrong_floating_parameter_dtype_after_assign(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "fp16.pt"
    quantize_checkpoint(source, output, "fp16")
    runtime = QuantizeRuntime(model_factory=CorruptAfterAssignTinyLLM)

    with pytest.raises(RuntimeError, match="after assign.*final_norm.weight"):
        load_quantized_checkpoint(output, runtime=runtime)


def test_direct_precision_loader_accepts_non_floating_buffers(tmp_path: Path) -> None:
    config = _model_config()
    model = IntegerBufferTinyLLM(config)
    source = tmp_path / "source-with-integer-buffer.pt"
    torch.save(
        {
            "version": 2,
            "model_state": model.state_dict(),
            "identity": {
                "model_config": config.model_dump(mode="json"),
                "tokenizer_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
            },
        },
        source,
    )
    output = tmp_path / "fp16.pt"
    runtime = QuantizeRuntime(model_factory=IntegerBufferTinyLLM)

    quantize_checkpoint(source, output, "fp16", runtime=runtime)
    loaded = load_quantized_checkpoint(output, runtime=runtime)

    assert loaded.model.quantization_marker.dtype is torch.int64
    assert loaded.model.quantization_marker.item() == 17


def test_loader_rejects_backend_configuration_mismatch(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "int8.pt"
    export_backend = HonestFakeQuantizationBackend()
    quantize_checkpoint(
        source,
        output,
        "int8-weight-only",
        runtime=_runtime(export_backend),
    )
    incompatible_backend = HonestFakeQuantizationBackend(bits_offset=1)

    with pytest.raises(RuntimeError, match="quantization configuration"):
        load_quantized_checkpoint(output, runtime=_runtime(incompatible_backend))


def test_optional_quantization_fails_explicitly_without_torchao(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "int8.pt"

    def unavailable():
        raise OptionalBackendUnavailableError("TorchAO quantization API is unavailable")

    monkeypatch.setattr(quantize_module, "_load_torchao_backend", unavailable)

    with pytest.raises(RuntimeError, match="int8-weight-only.*TorchAO"):
        quantize_checkpoint(source, output, "int8-weight-only")

    assert not output.exists()


def test_optional_quantization_does_not_swallow_backend_programming_error(
    tmp_path: Path,
) -> None:
    source = _checkpoint(tmp_path)
    unexpected = RuntimeError("programming bug")

    def broken_backend():
        raise unexpected

    with pytest.raises(RuntimeError) as raised:
        quantize_checkpoint(
            source,
            tmp_path / "int8.pt",
            "int8-weight-only",
            runtime=QuantizeRuntime(backend_loader=broken_backend),
        )

    assert raised.value is unexpected


def test_quantized_loader_maps_invalid_archive_to_checkpoint_format_error(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"not a torch checkpoint")

    with pytest.raises(CheckpointFormatError, match="invalid quantized checkpoint"):
        load_quantized_checkpoint(invalid)


def test_quantized_export_refuses_overwrite_by_default(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "export.pt"
    output.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError, match="already exists"):
        quantize_checkpoint(source, output, "bf16")

    assert output.read_bytes() == b"keep-me"


def test_quantized_export_can_atomically_replace_with_explicit_permission(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "export.pt"
    output.write_bytes(b"replace-me")

    manifest = quantize_checkpoint(source, output, "fp16", overwrite=True)

    loaded = load_quantized_checkpoint(output)
    assert loaded.manifest == manifest


def test_failed_export_leaves_no_output_or_temporary_file(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "broken.pt"

    def failing_save(payload: object, destination) -> None:
        destination.write(b"partial")
        raise OSError("disk full")

    runtime = QuantizeRuntime(save_payload=failing_save)

    with pytest.raises(OSError, match="disk full"):
        quantize_checkpoint(source, output, "bf16", runtime=runtime)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_failed_explicit_overwrite_preserves_existing_output(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    output = tmp_path / "existing.pt"
    output.write_bytes(b"original")

    def failing_save(payload: object, destination) -> None:
        destination.write(b"partial")
        raise OSError("disk full")

    runtime = QuantizeRuntime(save_payload=failing_save)

    with pytest.raises(OSError, match="disk full"):
        quantize_checkpoint(source, output, "bf16", overwrite=True, runtime=runtime)

    assert output.read_bytes() == b"original"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize("recipe", ["fp8", "int8", "automatic", ""])
def test_quantized_export_rejects_unknown_recipes(tmp_path: Path, recipe: str) -> None:
    source = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="recipe"):
        quantize_checkpoint(source, tmp_path / "output.pt", recipe)


def test_quantized_export_rejects_source_as_output_even_with_overwrite(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="source and output"):
        quantize_checkpoint(source, source, "bf16", overwrite=True)
