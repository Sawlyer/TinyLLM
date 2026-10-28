from pathlib import Path

import onnx
import onnxruntime as ort
import pytest
import torch

from tinyllm.cli import _build_parser
from tinyllm.config.schema import ModelConfig
from tinyllm.inference.onnx import export_onnx_forward
from tinyllm.model.transformer import TinyLLM


def create_model() -> TinyLLM:
    config = ModelConfig(
        vocab_size=16,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        mlp_ratio=2.0,
        dropout=0.0,
        rope_theta=10_000.0,
        rms_norm_eps=1e-5,
    )
    return TinyLLM(config).eval()


def test_onnx_forward_export_runs_in_runtime_for_dynamic_shapes(tmp_path: Path) -> None:
    model = create_model()
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    output = tmp_path / "model.onnx"

    report = export_onnx_forward(
        model,
        output,
        metadata={"tokenizer_sha256": "a" * 64},
        opset_version=18,
    )

    onnx.checker.check_model(str(output), full_check=True)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert session.get_inputs()[0].shape == ["batch", "sequence"]
    assert session.get_inputs()[0].type == "tensor(int64)"
    assert report.verified_shapes == ((1, 2), (2, 5))
    assert report.opset_version == 18
    assert all(
        torch.equal(original_state[name], value) for name, value in model.state_dict().items()
    )

    for shape in report.verified_shapes:
        input_ids = torch.randint(model.config.vocab_size, shape, dtype=torch.long)
        actual = session.run(["logits"], {"input_ids": input_ids.numpy()})[0]
        with torch.inference_mode():
            expected = model(input_ids).logits.numpy()
        assert actual.shape == (*shape, model.config.vocab_size)
        assert actual.dtype.name == "float32"
        torch.testing.assert_close(
            torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-5, rtol=1e-4
        )

    exported = onnx.load(str(output))
    properties = {item.key: item.value for item in exported.metadata_props}
    assert '"kv_cache":false' in properties["tinyllm.manifest"]
    assert '"tokenizer_sha256":"' + "a" * 64 + '"' in properties["tinyllm.manifest"]


def test_onnx_forward_export_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "model.onnx"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exists"):
        export_onnx_forward(create_model(), output, metadata={})

    assert output.read_bytes() == b"existing"


def test_onnx_cli_uses_checkpoint_and_separate_output_paths() -> None:
    args = _build_parser().parse_args(
        ["export-onnx", "--checkpoint", "base.pt", "--output", "model.onnx"]
    )

    assert args.handler.__name__ == "_export_onnx_command"
    assert args.checkpoint == Path("base.pt")
    assert args.output == Path("model.onnx")
