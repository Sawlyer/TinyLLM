"""Tokenizer and token-range compatibility checks at model boundaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

import numpy as np
from tokenizers import Tokenizer

from tinyllm.config.schema import ModelConfig
from tinyllm.data.artifacts import (
    REQUIRED_SPECIAL_TOKENS,
    CorpusManifest,
    open_token_artifact,
)


class TokenDatasetLike(Protocol):
    manifest: CorpusManifest
    split: str
    tokens: np.ndarray


def validate_manifest_for_model(
    manifest: CorpusManifest,
    model_config: ModelConfig,
    *,
    splits: Iterable[str] | None = None,
) -> None:
    """Reject tokenizer identities or corpus IDs outside model embedding rows."""
    if not isinstance(manifest, CorpusManifest):
        raise TypeError("manifest must be a CorpusManifest")
    _validate_identity(
        manifest.tokenizer_vocab_size,
        manifest.special_token_ids,
        model_config,
    )
    manifest_path = getattr(manifest, "_manifest_path", None)
    if not isinstance(manifest_path, Path):
        raise ValueError("manifest must be loaded from disk for corpus token validation")
    selected_splits = set(manifest.artifacts) if splits is None else set(splits)
    for split in selected_splits:
        if split not in manifest.artifacts:
            raise ValueError(f"unknown corpus split: {split}")
        tokens = open_token_artifact(manifest_path.parent, manifest.artifacts[split])
        _validate_token_range(tokens, model_config, split)


def validate_dataset_for_model(
    dataset: TokenDatasetLike,
    model_config: ModelConfig,
) -> None:
    """Validate an already-open dataset without requiring its manifest path."""
    _validate_identity(
        dataset.manifest.tokenizer_vocab_size,
        dataset.manifest.special_token_ids,
        model_config,
    )
    _validate_token_range(dataset.tokens, model_config, dataset.split)


def validate_tokenizer_for_model(
    tokenizer: Tokenizer,
    model_config: ModelConfig,
    *,
    expected_vocab_size: int | None = None,
    expected_special_ids: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Validate runtime tokenizer identity used by generation."""
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    actual_special_ids: dict[str, int] = {}
    for token in REQUIRED_SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token {token}")
        actual_special_ids[token] = token_id
    _validate_identity(actual_vocab_size, actual_special_ids, model_config)
    if expected_vocab_size is not None and actual_vocab_size != expected_vocab_size:
        raise ValueError("tokenizer vocabulary size differs from recorded identity")
    if expected_special_ids is not None and dict(expected_special_ids) != actual_special_ids:
        raise ValueError("tokenizer special IDs differ from recorded identity")
    return actual_special_ids


def _validate_identity(
    vocab_size: int | None,
    special_ids: Mapping[str, int] | None,
    model_config: ModelConfig,
) -> None:
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("tokenizer vocabulary identity is unavailable")
    if vocab_size > model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary {vocab_size} exceeds model vocab_size "
            f"{model_config.vocab_size}"
        )
    if not isinstance(special_ids, Mapping) or set(special_ids) != set(
        REQUIRED_SPECIAL_TOKENS
    ):
        raise ValueError("tokenizer special ID identity is incomplete")
    if len(set(special_ids.values())) != len(special_ids):
        raise ValueError("tokenizer special IDs must be distinct")
    for token, token_id in special_ids.items():
        if type(token_id) is not int or not 0 <= token_id < model_config.vocab_size:
            raise ValueError(
                f"special token {token} ID {token_id!r} exceeds model vocab_size "
                f"{model_config.vocab_size}"
            )


def _validate_token_range(
    tokens: np.ndarray,
    model_config: ModelConfig,
    split: str,
) -> None:
    if tokens.size == 0:
        return
    maximum = int(np.max(tokens))
    if maximum >= model_config.vocab_size:
        raise ValueError(
            f"corpus token {maximum} in {split} exceeds model vocab_size "
            f"{model_config.vocab_size}"
        )
