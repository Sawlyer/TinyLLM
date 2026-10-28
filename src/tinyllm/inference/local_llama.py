"""Strict local-only adapter for Hugging Face LlamaForCausalLM weights."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from tinyllm.config.schema import ModelConfig
from tinyllm.data.validation import validate_tokenizer_for_model
from tinyllm.model.transformer import TinyLLM


@dataclass(frozen=True)
class LocalLlamaCheckpoint:
    model: TinyLLM
    model_config: ModelConfig
    tokenizer: Tokenizer
    tokenizer_sha256: str
    manifest: dict[str, Any]


def _require_positive_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"Llama config field {key!r} must be a positive integer")
    return value


def _model_config(external: dict[str, Any]) -> ModelConfig:
    architecture = external.get("architectures")
    if external.get("model_type") != "llama" or architecture != ["LlamaForCausalLM"]:
        raise ValueError(
            "supported architecture is model_type='llama' with "
            "architectures=['LlamaForCausalLM']"
        )
    if external.get("rope_scaling") is not None:
        raise ValueError("Llama rope_scaling is unsupported; use standard RoPE")
    if external.get("hidden_act", "silu") != "silu":
        raise ValueError("Llama hidden_act must be 'silu'")
    if external.get("attention_bias", False) is not False:
        raise ValueError("Llama attention_bias must be false")
    if external.get("mlp_bias", False) is not False:
        raise ValueError("Llama mlp_bias must be false")
    if external.get("pretraining_tp", 1) != 1:
        raise ValueError("Llama pretraining_tp must be 1")
    tie_word_embeddings = external.get("tie_word_embeddings", False)
    if type(tie_word_embeddings) is not bool:
        raise ValueError("Llama tie_word_embeddings must be a boolean")

    hidden_size = _require_positive_int(external, "hidden_size")
    n_heads = _require_positive_int(external, "num_attention_heads")
    head_dim = external.get("head_dim", hidden_size // n_heads)
    if type(head_dim) is not int or head_dim * n_heads != hidden_size or head_dim % 2:
        raise ValueError(
            "Llama head_dim must be even and match hidden_size / num_attention_heads"
        )
    intermediate_size = _require_positive_int(external, "intermediate_size")
    ratio = intermediate_size / hidden_size
    model_config = ModelConfig(
        vocab_size=_require_positive_int(external, "vocab_size"),
        max_seq_len=_require_positive_int(external, "max_position_embeddings"),
        d_model=hidden_size,
        n_layers=_require_positive_int(external, "num_hidden_layers"),
        n_heads=n_heads,
        n_kv_heads=external.get("num_key_value_heads", n_heads),
        mlp_ratio=ratio,
        dropout=0.0,
        rope_theta=external.get("rope_theta", 10000.0),
        rms_norm_eps=external.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=tie_word_embeddings,
    )
    if int(model_config.d_model * model_config.mlp_ratio) != intermediate_size:
        raise ValueError("Llama intermediate_size cannot be represented by TinyLLM mlp_ratio")
    return model_config


def _local_file(root: Path, filename: str) -> Path:
    path = root / filename
    if path.is_symlink():
        raise ValueError(f"model file must not be a symlink: {filename}")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"model file must resolve inside the selected local directory: {filename}")
    return resolved


def _weight_name_map(n_layers: int) -> dict[str, str]:
    names = {
        "model.embed_tokens.weight": "token_embedding.weight",
        "model.norm.weight": "final_norm.weight",
    }
    layer_names = {
        "input_layernorm.weight": "attention_norm.weight",
        "self_attn.q_proj.weight": "attention.q_proj.weight",
        "self_attn.k_proj.weight": "attention.k_proj.weight",
        "self_attn.v_proj.weight": "attention.v_proj.weight",
        "self_attn.o_proj.weight": "attention.out_proj.weight",
        "post_attention_layernorm.weight": "feed_forward_norm.weight",
        "mlp.gate_proj.weight": "feed_forward.gate_proj.weight",
        "mlp.up_proj.weight": "feed_forward.up_proj.weight",
        "mlp.down_proj.weight": "feed_forward.down_proj.weight",
    }
    for index in range(n_layers):
        for source_suffix, target_suffix in layer_names.items():
            names[f"model.layers.{index}.{source_suffix}"] = f"blocks.{index}.{target_suffix}"
    names["lm_head.weight"] = "lm_head.weight"
    return names


def load_local_llama(
    model_directory: Path,
    *,
    device: torch.device | str = "cpu",
) -> LocalLlamaCheckpoint:
    """Load one unsharded Llama checkpoint using files from an explicit local directory."""
    root = Path(model_directory).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("model directory must be a local directory")
    if (root / "model.safetensors.index.json").exists():
        raise ValueError("sharded Llama checkpoints are unsupported; provide one model.safetensors")

    config_path = _local_file(root, "config.json")
    tokenizer_path: Path
    weights_path: Path
    try:
        external_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Llama config.json is not valid UTF-8 JSON") from error
    if not isinstance(external_config, dict):
        raise ValueError("Llama config.json must contain a JSON object")
    model_config = _model_config(external_config)
    tokenizer_path = _local_file(root, "tokenizer.json")
    weights_path = _local_file(root, "model.safetensors")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    special_ids = validate_tokenizer_for_model(tokenizer, model_config)

    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError(
            "local Llama loading requires safetensors; install tinyllm[external]"
        ) from error
    source_state = load_file(str(weights_path), device="cpu")
    model = TinyLLM(model_config)
    name_map = _weight_name_map(model_config.n_layers)
    expected_state = model.state_dict()
    optional_tied_head = model_config.tie_word_embeddings
    expected_source_names = set(name_map)
    if optional_tied_head:
        expected_source_names.remove("lm_head.weight")
    source_names = set(source_state)
    missing = expected_source_names - source_names
    unexpected = source_names - set(name_map)
    if missing or unexpected:
        raise ValueError(
            f"Llama tensor names are incompatible; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    converted: dict[str, torch.Tensor] = {}
    for source_name, tensor in source_state.items():
        if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
            raise ValueError(f"Llama tensor must use a floating dtype: {source_name}")
        target_name = name_map[source_name]
        if target_name == "lm_head.weight" and optional_tied_head:
            if not torch.equal(tensor, source_state["model.embed_tokens.weight"]):
                raise ValueError("tied Llama lm_head.weight differs from model.embed_tokens.weight")
        if target_name in expected_state and tensor.shape != expected_state[target_name].shape:
            raise ValueError(
                f"Llama tensor shape mismatch for {source_name}: "
                f"expected {tuple(expected_state[target_name].shape)}, got {tuple(tensor.shape)}"
            )
        converted[target_name] = tensor
    if optional_tied_head:
        converted["lm_head.weight"] = converted["token_embedding.weight"]
    if set(converted) != set(expected_state):
        raise ValueError("Llama tensor mapping does not cover the TinyLLM model state")
    model.load_state_dict(converted, strict=True)
    model.eval()
    model.to(device=torch.device(device))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return LocalLlamaCheckpoint(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        tokenizer_sha256=digest(tokenizer_path),
        manifest={
            "architecture": "LlamaForCausalLM",
            "model_type": "llama",
            "format": "safetensors",
            "model_config": model_config.model_dump(mode="json"),
            "special_token_ids": special_ids,
            "files": {
                "config.json": {"sha256": digest(config_path)},
                "tokenizer.json": {"sha256": digest(tokenizer_path)},
                "model.safetensors": {"sha256": digest(weights_path)},
            },
            "remote_code_executed": False,
            "network_accessed": False,
        },
    )
