"""Deterministic, read-only diagnostics for prepared tokenizer artifacts."""

from __future__ import annotations

import hashlib
import tempfile
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from tinyllm.config.schema import TokenizerConfig
from tinyllm.data.artifacts import load_manifest, open_token_artifact
from tinyllm.data.tokenizer import train_tokenizer


def diagnose_tokenizer(
    manifest_path: Path,
    *,
    splits: Iterable[str] = ("train", "validation"),
    sample_documents: int = 1000,
    compare_vocab_sizes: Iterable[int] = (),
) -> dict[str, Any]:
    """Summarize tokenizer and corpus behavior without modifying source artifacts."""
    if type(sample_documents) is not int or sample_documents <= 0:
        raise ValueError("sample_documents must be a positive integer")
    selected_splits = tuple(dict.fromkeys(splits))
    if not selected_splits:
        raise ValueError("at least one split must be selected")

    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path, splits=selected_splits)
    tokenizer_path = Path(manifest.tokenizer_path)
    if not tokenizer_path.is_absolute():
        tokenizer_path = manifest_path.parent / tokenizer_path
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    eos_id = manifest.special_token_ids["<eos>"]
    unk_id = tokenizer.token_to_id("<unk>")
    tokenizer_config = TokenizerConfig.model_validate(manifest.tokenizer_config)

    reports: dict[str, Any] = {}
    training_sample: list[str] = []
    for split in selected_splits:
        artifact = manifest.artifacts[split]
        token_ids = open_token_artifact(manifest_path.parent, artifact)
        token_counts = _count_token_ids(token_ids, vocab_size)
        document_lengths, sampled_documents = _collect_documents(
            token_ids,
            eos_id=eos_id,
            limit=sample_documents,
        )
        decoded = [
            tokenizer.decode(document, skip_special_tokens=False)
            for document in sampled_documents
        ]
        report = _split_report(
            split=split,
            artifact=artifact,
            document_lengths=document_lengths,
            sampled_documents=sampled_documents,
            decoded_documents=decoded,
            token_counts=token_counts,
            tokenizer=tokenizer,
            unk_id=unk_id,
            vocab_size=vocab_size,
            special_token_ids=manifest.special_token_ids,
            declared_documents=manifest.document_counts[split],
            sample_limit=sample_documents,
        )
        reports[split] = report
        if split == "train":
            training_sample = decoded

    comparisons = []
    if compare_vocab_sizes:
        if not training_sample:
            raise ValueError("vocabulary comparisons require the train split")
        for requested_size in dict.fromkeys(compare_vocab_sizes):
            minimum_size = len(tokenizer_config.special_tokens) + 1
            if type(requested_size) is not int or not minimum_size <= requested_size <= 65536:
                raise ValueError(
                    "vocabulary sizes must be integers between "
                    f"{minimum_size} and 65536"
                )
            candidate = tokenizer
            if requested_size != vocab_size:
                candidate_config = tokenizer_config.model_copy(
                    update={"vocab_size": requested_size}
                )
                with tempfile.TemporaryDirectory(prefix="tinyllm-tokenizer-diagnostic-") as root:
                    candidate = train_tokenizer(
                        training_sample,
                        candidate_config,
                        Path(root) / "tokenizer.json",
                    )
            comparisons.append(
                _compare_tokenizer(
                    candidate,
                    training_sample,
                    requested_vocab_size=requested_size,
                )
            )

    return {
        "manifest": str(manifest_path.resolve()),
        "dataset": manifest.dataset_name,
        "dataset_revision": manifest.dataset_revision,
        "tokenizer": {
            "path": str(tokenizer_path.resolve()),
            "sha256": manifest.tokenizer_sha256,
            "vocabulary_size": vocab_size,
            "requested_vocabulary_size": tokenizer_config.vocab_size,
            "minimum_frequency": tokenizer_config.min_frequency,
            "special_tokens": {
                token: {
                    "id": token_id,
                }
                for token, token_id in manifest.special_token_ids.items()
            },
        },
        "sample_documents_limit": sample_documents,
        "splits": reports,
        "vocabulary_comparisons": comparisons,
    }


def _count_token_ids(token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.uint64)
    chunk_size = 1_000_000
    for start in range(0, len(token_ids), chunk_size):
        chunk = np.asarray(token_ids[start : start + chunk_size], dtype=np.int64)
        if chunk.size and (int(chunk.min()) < 0 or int(chunk.max()) >= vocab_size):
            raise ValueError("token artifact contains an ID outside tokenizer vocabulary")
        counts += np.bincount(chunk, minlength=vocab_size).astype(np.uint64, copy=False)
    return counts


def _collect_documents(
    token_ids: np.ndarray,
    *,
    eos_id: int,
    limit: int,
) -> tuple[list[int], list[list[int]]]:
    lengths: list[int] = []
    sampled: list[list[int]] = []
    current: list[int] = []
    for value in token_ids:
        token_id = int(value)
        if token_id == eos_id:
            lengths.append(len(current))
            if len(sampled) < limit:
                sampled.append(current)
            current = []
        else:
            current.append(token_id)
    if current:
        lengths.append(len(current))
        if len(sampled) < limit:
            sampled.append(current)
    return lengths, sampled


def _split_report(
    *,
    split: str,
    artifact: Any,
    document_lengths: list[int],
    sampled_documents: list[list[int]],
    decoded_documents: list[str],
    token_counts: np.ndarray,
    tokenizer: Tokenizer,
    unk_id: int | None,
    vocab_size: int,
    special_token_ids: dict[str, int],
    declared_documents: int,
    sample_limit: int,
) -> dict[str, Any]:
    sampled_token_total = sum(len(document) for document in sampled_documents)
    sampled_unknown_tokens = (
        sum(document.count(unk_id) for document in sampled_documents)
        if unk_id is not None
        else 0
    )
    sampled_text = "".join(decoded_documents)
    seen: set[str] = set()
    duplicate_documents = 0
    for text in decoded_documents:
        fingerprint = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
        if fingerprint in seen:
            duplicate_documents += 1
        seen.add(fingerprint)

    control_chars = 0
    format_chars = 0
    replacement_chars = 0
    for character in sampled_text:
        category = unicodedata.category(character)
        if category == "Cc" and character not in "\n\r\t":
            control_chars += 1
        elif category == "Cf":
            format_chars += 1
        if character == "\ufffd" or category == "Cs" or _is_unicode_noncharacter(character):
            replacement_chars += 1

    top_ids = sorted(
        range(vocab_size), key=lambda token_id: (-int(token_counts[token_id]), token_id)
    )[:20]
    top_tokens = [
        {
            "id": int(token_id),
            "token": tokenizer.id_to_token(int(token_id)),
            "frequency": int(token_counts[token_id]),
        }
        for token_id in top_ids
        if token_counts[token_id]
    ]
    lengths_sorted = sorted(document_lengths)
    warnings = []
    if len(document_lengths) != declared_documents:
        warnings.append("observed document count differs from manifest")
    if any(not text for text in decoded_documents):
        warnings.append("empty documents found in inspected sample")
    if duplicate_documents:
        warnings.append("duplicate documents found in inspected sample")
    if replacement_chars:
        warnings.append("invalid or replacement Unicode characters found in inspected sample")
    if control_chars:
        warnings.append("unexpected control characters found in inspected sample")
    if format_chars:
        warnings.append("Unicode format characters found in inspected sample")
    return {
        "artifact": {
            "filename": artifact.filename,
            "sha256": artifact.sha256,
            "tokenizer_sha256": artifact.tokenizer_sha256,
            "token_count": artifact.token_count,
        },
        "documents": {
            "declared": declared_documents,
            "observed": len(document_lengths),
            "sampled": len(decoded_documents),
            "empty_sampled": sum(not text for text in decoded_documents),
            "duplicate_sampled": duplicate_documents,
            "mean_tokens": _mean(document_lengths),
            "median_tokens": _percentile(lengths_sorted, 50),
            "p90_tokens": _percentile(lengths_sorted, 90),
            "max_tokens": max(document_lengths, default=0),
        },
        "sample": {
            "method": "first documents in split, bounded by sample_documents_limit",
            "limit": sample_limit,
            "characters": len(sampled_text),
            "tokens": sampled_token_total,
            "characters_per_token": len(sampled_text) / sampled_token_total
            if sampled_token_total
            else 0.0,
            "unknown_tokens": sampled_unknown_tokens,
            "unknown_tokens_total": int(token_counts[unk_id]) if unk_id is not None else 0,
            "unicode_replacement_or_invalid_chars": replacement_chars,
            "control_chars": control_chars,
            "format_chars": format_chars,
            "examples": [text[:300] for text in decoded_documents[:5]],
        },
        "warnings": warnings,
        "vocabulary": {
            "size": vocab_size,
            "used_ids": int(np.count_nonzero(token_counts)),
            "unused_ids": int(vocab_size - np.count_nonzero(token_counts)),
            "special_token_frequencies": {
                token: int(token_counts[token_id]) for token, token_id in special_token_ids.items()
            },
            "top_tokens": top_tokens,
        },
    }


def _compare_tokenizer(
    tokenizer: Tokenizer,
    texts: list[str],
    *,
    requested_vocab_size: int,
) -> dict[str, Any]:
    characters = 0
    token_count = 0
    unknown_tokens = 0
    unk_id = tokenizer.token_to_id("<unk>")
    for text in texts:
        encoded = tokenizer.encode(text)
        characters += len(text)
        token_count += len(encoded.ids)
        if unk_id is not None:
            unknown_tokens += encoded.ids.count(unk_id)
    return {
        "requested_vocabulary_size": requested_vocab_size,
        "actual_vocabulary_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "documents": len(texts),
        "characters": characters,
        "tokens": token_count,
        "characters_per_token": characters / token_count if token_count else 0.0,
        "unknown_tokens": unknown_tokens,
    }


def _is_unicode_noncharacter(character: str) -> bool:
    codepoint = ord(character)
    return 0xFDD0 <= codepoint <= 0xFDEF or (codepoint & 0xFFFF) in {0xFFFE, 0xFFFF}


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[int], percentile: int) -> int:
    if not sorted_values:
        return 0
    index = min((len(sorted_values) - 1) * percentile // 100, len(sorted_values) - 1)
    return sorted_values[index]

