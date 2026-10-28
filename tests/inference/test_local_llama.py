import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models

from tinyllm.config.schema import ModelConfig
from tinyllm.inference.local_llama import load_local_llama
from tinyllm.model.transformer import TinyLLM


def _config(*, architecture="LlamaForCausalLM", tie_word_embeddings=True):
    return {
        "architectures": [architecture],
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 4,
        "max_position_embeddings": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "vocab_size": 4,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": tie_word_embeddings,
        "rope_scaling": None,
        "attention_bias": False,
        "mlp_bias": False,
        "head_dim": 4,
    }


def _write_local_llama(
    directory: Path,
    *,
    architecture="LlamaForCausalLM",
    tie_word_embeddings=True,
    mismatch=False,
):
    safetensors = pytest.importorskip("safetensors.torch")
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            _config(
                architecture=architecture,
                tie_word_embeddings=tie_word_embeddings,
            )
        )
    )
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.save(str(directory / "tokenizer.json"))
    tiny_config = ModelConfig(
        vocab_size=4,
        max_seq_len=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        mlp_ratio=0.5,
        dropout=0.0,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=tie_word_embeddings,
    )
    state = TinyLLM(tiny_config).state_dict()
    names = {
        "model.embed_tokens.weight": "token_embedding.weight",
        "model.layers.0.input_layernorm.weight": "blocks.0.attention_norm.weight",
        "model.layers.0.self_attn.q_proj.weight": "blocks.0.attention.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight": "blocks.0.attention.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight": "blocks.0.attention.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight": "blocks.0.attention.out_proj.weight",
        "model.layers.0.post_attention_layernorm.weight": "blocks.0.feed_forward_norm.weight",
        "model.layers.0.mlp.gate_proj.weight": "blocks.0.feed_forward.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight": "blocks.0.feed_forward.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight": "blocks.0.feed_forward.down_proj.weight",
        "model.norm.weight": "final_norm.weight",
    }
    if not tie_word_embeddings:
        names["lm_head.weight"] = "lm_head.weight"
    external = {key: state[value].clone() for key, value in names.items()}
    if mismatch:
        external["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(7, 8)
    safetensors.save_file(external, str(directory / "model.safetensors"))


def test_local_llama_loads_supported_weights_and_reports_local_identity(tmp_path):
    model_dir = tmp_path / "llama"
    _write_local_llama(model_dir)

    imported = load_local_llama(model_dir)

    assert imported.model_config.d_model == 8
    assert imported.model_config.n_kv_heads == 1
    assert imported.manifest["architecture"] == "LlamaForCausalLM"
    assert imported.manifest["format"] == "safetensors"
    assert imported.tokenizer_sha256
    assert imported.model.token_embedding.weight is imported.model.lm_head.weight
    assert imported.model.training is False


def test_local_llama_preserves_untied_output_weights(tmp_path):
    model_dir = tmp_path / "untied"
    _write_local_llama(model_dir, tie_word_embeddings=False)

    imported = load_local_llama(model_dir)

    assert imported.model_config.tie_word_embeddings is False
    assert imported.model.token_embedding.weight is not imported.model.lm_head.weight


def test_local_llama_rejects_architecture_before_reading_weights(tmp_path):
    model_dir = tmp_path / "unsupported"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(_config(architecture="MistralForCausalLM")))

    with pytest.raises(ValueError, match="supported architecture.*LlamaForCausalLM"):
        load_local_llama(model_dir)


def test_local_llama_rejects_incompatible_tensor_shape(tmp_path):
    model_dir = tmp_path / "bad-shape"
    _write_local_llama(model_dir, mismatch=True)

    with pytest.raises(ValueError, match="shape mismatch.*q_proj"):
        load_local_llama(model_dir)


def test_model_config_comparison_accepts_legacy_omitted_tying_field():
    from tinyllm import cli

    configured = ModelConfig(
        vocab_size=4,
        max_seq_len=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        mlp_ratio=0.5,
        dropout=0.0,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    )
    legacy = configured.model_dump(mode="json")
    legacy.pop("tie_word_embeddings")

    cli._require_model_config_match(legacy, configured)


def test_compare_local_llama_cli_accepts_local_inputs(tmp_path):
    from tinyllm.cli import _build_parser

    args = _build_parser().parse_args(
        [
            "compare-local-llama",
            "--checkpoint",
            str(tmp_path / "source.pt"),
            "--model-dir",
            str(tmp_path / "local-llama"),
            "--tokens",
            "64",
        ]
    )

    assert args.split == "validation"
    assert args.tokens == 64
    assert args.handler.__name__ == "_compare_local_llama_command"


def test_compare_local_llama_command_writes_named_external_comparison(
    monkeypatch, tmp_path
):
    from tinyllm import cli

    checkpoint = tmp_path / "reference.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    output = tmp_path / "comparison.json"
    checkpoint.write_bytes(b"checkpoint")
    tokenizer_path.write_bytes(b"tokenizer")
    (tmp_path / "llama").mkdir()
    (tmp_path / "llama" / "model.safetensors").write_bytes(b"weights")
    model_config = ModelConfig(
        vocab_size=4,
        max_seq_len=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        mlp_ratio=0.5,
        dropout=0.0,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    corpus = SimpleNamespace(
        artifacts={"validation": SimpleNamespace(sha256="c" * 64)}
    )
    config = SimpleNamespace(
        model=model_config,
        tokenizer=SimpleNamespace(path=tokenizer_path),
        training=SimpleNamespace(device="cpu", micro_batch_size=2),
    )
    reference = SimpleNamespace(
        model=object(),
        identity=SimpleNamespace(
            model_config=model_config.model_dump(mode="json"),
            tokenizer_sha256="d" * 64,
        ),
    )
    external = SimpleNamespace(
        model=object(),
        model_config=model_config,
        tokenizer_sha256="d" * 64,
        manifest={
            "files": {"model.safetensors": {"sha256": "e" * 64}},
        },
    )
    pair = {
        "schema_version": 1,
        "evaluation": {"evaluated_tokens": 12},
        "reference": {"mean_loss": 1.0},
        "quantized": {
            "mean_loss": 1.1,
            "perplexity": 3.0,
            "latency_ratio": 1.2,
        },
        "quality": {
            "loss_delta_nats_per_token": 0.1,
            "perplexity_ratio": 1.1,
            "latency_ratio": 1.2,
        },
    }
    monkeypatch.setattr(cli, "_load_args_config", lambda _args: config)
    monkeypatch.setattr(cli, "_validated_manifest", lambda *_args, **_kwargs: corpus)
    monkeypatch.setattr(cli, "_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "_resolve_inference_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr(cli, "load_model_checkpoint", lambda *_args, **_kwargs: reference)
    monkeypatch.setattr(cli, "load_local_llama", lambda *_args, **_kwargs: external)
    monkeypatch.setattr(
        cli,
        "sha256_file",
        lambda path: "d" * 64 if Path(path) == tokenizer_path else "f" * 64,
    )
    monkeypatch.setattr(
        cli,
        "compare_quantized_models",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: pair),
    )
    written = {}
    monkeypatch.setattr(
        cli,
        "write_json_report",
        lambda report, _path, **_kwargs: written.update(report),
    )
    args = SimpleNamespace(
        checkpoint=checkpoint,
        model_dir=tmp_path / "llama",
        output=output,
        config=None,
        seed=42,
        tokens=12,
        batches=None,
        split="validation",
        device=None,
        overwrite=False,
        warmup_runs=0,
        measured_runs=2,
    )

    cli._compare_local_llama_command(args)

    assert written["comparison_type"] == "tinyllm-vs-local-llama"
    assert written["external_model"]["mean_loss"] == 1.1
    assert written["comparison"]["loss_delta_nats_per_token"] == 0.1
    assert "quantized" not in written
