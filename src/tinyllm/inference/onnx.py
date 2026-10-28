"""Static-shape-independent ONNX export for TinyLLM forward inference."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tinyllm.model.transformer import TinyLLM


@dataclass(frozen=True, slots=True)
class OnnxExportReport:
    output_path: Path
    opset_version: int
    verified_shapes: tuple[tuple[int, int], ...]
    cpu_provider: str
    cuda_verified: bool


class _ForwardLogits(nn.Module):
    def __init__(self, model: TinyLLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor) -> Tensor:
        output = self.model(input_ids)
        if output.logits is None:
            raise RuntimeError("model forward returned no logits")
        return output.logits


def export_onnx_forward(
    model: TinyLLM,
    output_path: Path,
    *,
    metadata: Mapping[str, Any],
    opset_version: int = 18,
    overwrite: bool = False,
) -> OnnxExportReport:
    """Export logits-only FP32 inference graph and verify it with ONNX Runtime."""
    output_path = Path(output_path)
    if output_path.suffix.casefold() != ".onnx":
        raise ValueError("ONNX output path must end in .onnx")
    if type(opset_version) is not int or opset_version != 18:
        raise ValueError("TinyLLM ONNX export currently supports opset 18 only")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"ONNX output already exists: {output_path}")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("ONNX metadata must contain finite JSON-compatible values") from error

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import onnxscript  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "install the optional `onnx` dependencies to export this model"
        ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.stem}.",
            suffix=".onnx",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        temporary_path.unlink()

        export_model = copy.deepcopy(model).to(device="cpu", dtype=torch.float32).eval()
        adapter = _ForwardLogits(export_model).eval()
        max_seq_len = export_model.config.max_seq_len
        vocab_size = export_model.config.vocab_size
        sequence_lengths = (min(2, max_seq_len), min(5, max_seq_len))
        verified_shapes = ((1, sequence_lengths[0]), (2, sequence_lengths[1]))
        example_input = torch.randint(
            vocab_size,
            verified_shapes[0],
            dtype=torch.long,
            generator=torch.Generator(device="cpu").manual_seed(17),
        )
        batch_dimension = torch.export.Dim("batch", min=1)
        sequence_dimension = torch.export.Dim("sequence", min=1, max=max_seq_len)
        torch.onnx.export(
            adapter,
            (example_input,),
            f=temporary_path,
            input_names=["input_ids"],
            output_names=["logits"],
            opset_version=opset_version,
            dynamo=True,
            dynamic_shapes={
                "input_ids": {0: batch_dimension, 1: sequence_dimension},
            },
        )
        onnx.checker.check_model(str(temporary_path), full_check=True)

        cpu_session = ort.InferenceSession(
            str(temporary_path), providers=["CPUExecutionProvider"]
        )
        _verify_runtime_shapes(cpu_session, adapter, verified_shapes, vocab_size, np, torch)

        cuda_verified = False
        if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
            cuda_session = ort.InferenceSession(
                str(temporary_path), providers=["CUDAExecutionProvider"]
            )
            _verify_runtime_shapes(cuda_session, adapter, verified_shapes, vocab_size, np, torch)
            cuda_verified = True

        manifest = {
            "format_version": 1,
            "framework": "PyTorch",
            "runtime": "ONNX Runtime",
            "opset_version": opset_version,
            "input": {"name": "input_ids", "dtype": "int64", "shape": ["batch", "sequence"]},
            "output": {
                "name": "logits",
                "dtype": "float32",
                "shape": ["batch", "sequence", vocab_size],
            },
            "constraints": {
                "batch_min": 1,
                "sequence_min": 1,
                "sequence_max": max_seq_len,
            },
            "model_config": export_model.config.model_dump(mode="json"),
            "kv_cache": False,
            "tokenizer": "external; input IDs must use tokenizer recorded in model metadata",
            "verified_shapes": [list(shape) for shape in verified_shapes],
            "runtime_validation": {
                "cpu_provider": "CPUExecutionProvider",
                "cpu_verified": True,
                "cuda_verified": cuda_verified,
            },
            "source": dict(metadata),
        }
        exported_model = onnx.load(str(temporary_path))
        property_entry = exported_model.metadata_props.add()
        property_entry.key = "tinyllm.manifest"
        property_entry.value = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        onnx.save(exported_model, str(temporary_path))
        onnx.checker.check_model(str(temporary_path), full_check=True)
        os.replace(temporary_path, output_path)
        temporary_path = None
        return OnnxExportReport(
            output_path=output_path,
            opset_version=opset_version,
            verified_shapes=verified_shapes,
            cpu_provider="CPUExecutionProvider",
            cuda_verified=cuda_verified,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _verify_runtime_shapes(
    session: Any,
    model: nn.Module,
    shapes: tuple[tuple[int, int], ...],
    vocab_size: int,
    numpy: Any,
    torch_module: Any,
) -> None:
    for batch_size, sequence_length in shapes:
        input_ids = torch_module.randint(
            vocab_size,
            (batch_size, sequence_length),
            dtype=torch_module.long,
            generator=torch_module.Generator(device="cpu").manual_seed(
                17 + batch_size + sequence_length
            ),
        )
        runtime_logits = session.run(
            ["logits"], {"input_ids": input_ids.numpy()}
        )[0]
        with torch_module.inference_mode():
            reference_logits = model(input_ids).detach().cpu().numpy()
        if runtime_logits.dtype != numpy.float32:
            raise RuntimeError("ONNX Runtime returned logits outside the documented FP32 dtype")
        numpy.testing.assert_allclose(runtime_logits, reference_logits, rtol=1e-4, atol=1e-5)
