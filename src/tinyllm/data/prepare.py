"""End-to-end deterministic corpus preparation."""

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tinyllm.config.schema import TinyLLMConfig
from tinyllm.data.artifacts import (
    MANIFEST_VERSION,
    REQUIRED_SPECIAL_TOKENS,
    CorpusManifest,
    TokenArtifactManifest,
    load_manifest,
    open_token_artifact,
    sha256_file,
    write_manifest,
    write_token_chunks,
)
from tinyllm.data.clean import normalize_document
from tinyllm.data.split import split_for_id
from tinyllm.data.tokenizer import train_tokenizer

SPLITS = ("train", "validation")


def prepare_corpus(
    config: TinyLLMConfig,
    overwrite: bool = False,
    documents: Iterable[Mapping[str, Any] | str] | None = None,
) -> CorpusManifest:
    """Prepare a checksummed corpus from injected or streaming source documents."""
    output_dir = config.data.output_dir.resolve()
    tokenizer_path = config.tokenizer.path.resolve()
    manifest_path = output_dir / "manifest.json"
    final_paths = [
        output_dir / "train.bin",
        output_dir / "validation.bin",
        tokenizer_path,
        manifest_path,
    ]
    existing = [path for path in final_paths if path.exists()]
    if existing and not overwrite:
        missing = [path for path in final_paths if not path.exists()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileExistsError(
                f"corpus outputs incomplete; pass overwrite=True; missing: {names}"
            )
        try:
            manifest = load_manifest(manifest_path)
        except ValueError as error:
            raise FileExistsError(
                f"corpus outputs invalid; pass overwrite=True: {error}"
            ) from error
        incompatibility = _manifest_incompatibility(config, manifest)
        if incompatibility is not None:
            raise FileExistsError(
                f"corpus outputs incompatible with config; pass overwrite=True: {incompatibility}"
            )
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    staging_dir = output_dir / f".prepare-{transaction_id}.tmp"
    staged_tokenizer = tokenizer_path.with_name(f".{tokenizer_path.name}.{transaction_id}.tmp")
    staging_dir.mkdir()
    try:
        if documents is None:
            source, resolved_revision, upstream_fingerprint = _stream_hugging_face(config)
        else:
            source = documents
            resolved_revision = None
            upstream_fingerprint = None
        document_counts, content_fingerprint = _spool_documents(source, config, staging_dir)
        if document_counts["train"] == 0:
            raise ValueError("training split contains no usable documents")

        dataset_revision = resolved_revision or f"injected-sha256:{content_fingerprint}"
        dataset_fingerprint = upstream_fingerprint or content_fingerprint

        tokenizer = train_tokenizer(
            _read_spool(staging_dir / "train.documents.jsonl"),
            config.tokenizer,
            staged_tokenizer,
        )
        tokenizer_sha256 = sha256_file(staged_tokenizer)
        tokenizer_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
        special_token_ids = {
            token: tokenizer.token_to_id(token) for token in REQUIRED_SPECIAL_TOKENS
        }
        if any(token_id is None for token_id in special_token_ids.values()):
            raise ValueError("trained tokenizer is missing required special tokens")
        resolved_special_ids = {
            token: int(token_id) for token, token_id in special_token_ids.items()
        }
        eos_id = resolved_special_ids["<eos>"]

        artifacts: dict[str, TokenArtifactManifest] = {}
        for split in SPLITS:
            texts = _read_spool(staging_dir / f"{split}.documents.jsonl")
            artifacts[split] = write_token_chunks(
                staging_dir,
                split,
                _encode_documents(texts, tokenizer, eos_id),
                tokenizer_sha256=tokenizer_sha256,
            )
            open_token_artifact(staging_dir, artifacts[split])

        manifest = CorpusManifest(
            version=MANIFEST_VERSION,
            dataset_name=config.data.dataset_name,
            dataset_requested_revision=config.data.dataset_revision,
            dataset_revision=dataset_revision,
            dataset_fingerprint=dataset_fingerprint,
            text_field=config.data.text_field,
            seed=config.data.seed,
            validation_fraction=config.data.validation_fraction,
            tokenizer_path=str(tokenizer_path.resolve()),
            tokenizer_sha256=tokenizer_sha256,
            tokenizer_config=config.tokenizer.model_dump(mode="json"),
            data_config=config.data.model_dump(mode="json"),
            document_counts=document_counts,
            artifacts=artifacts,
            tokenizer_vocab_size=tokenizer_vocab_size,
            special_token_ids=resolved_special_ids,
        )
        staged_manifest = staging_dir / "manifest.json"
        write_manifest(staged_manifest, manifest)
        promotions = [
            (staging_dir / "train.bin", output_dir / "train.bin"),
            (staging_dir / "validation.bin", output_dir / "validation.bin"),
            (staged_tokenizer, tokenizer_path),
            (staged_manifest, manifest_path),
        ]
        _promote_transaction(promotions, transaction_id)
        return load_manifest(manifest_path)
    finally:
        staged_tokenizer.unlink(missing_ok=True)
        shutil.rmtree(staging_dir, ignore_errors=True)


def _manifest_incompatibility(
    config: TinyLLMConfig,
    manifest: CorpusManifest,
) -> str | None:
    if manifest.data_config != config.data.model_dump(mode="json"):
        return "data configuration differs"
    if manifest.tokenizer_config != config.tokenizer.model_dump(mode="json"):
        return "tokenizer configuration differs"
    if manifest.dataset_name != config.data.dataset_name:
        return "dataset name differs"
    if manifest.dataset_requested_revision != config.data.dataset_revision:
        return "requested dataset revision differs"
    if Path(manifest.tokenizer_path).resolve() != config.tokenizer.path.resolve():
        return "tokenizer path differs"
    return None


def _stream_hugging_face(
    config: TinyLLMConfig,
) -> tuple[Iterable[Mapping[str, Any]], str, str | None]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    requested_revision = config.data.dataset_revision
    if _is_commit_sha(requested_revision):
        resolved_revision = requested_revision
    else:
        dataset_info = HfApi().dataset_info(
            config.data.dataset_name,
            revision=requested_revision,
        )
        resolved_revision = dataset_info.sha
    if not _is_commit_sha(resolved_revision):
        raise ValueError("Hugging Face dataset revision did not resolve to a commit SHA")

    dataset = load_dataset(
        config.data.dataset_name,
        revision=resolved_revision,
        split="train",
        streaming=True,
        cache_dir=str(config.data.cache_dir),
    )
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        fingerprint = None
    return dataset, resolved_revision, fingerprint


def _spool_documents(
    documents: Iterable[Mapping[str, Any] | str],
    config: TinyLLMConfig,
    staging_dir: Path,
) -> tuple[dict[str, int], str]:
    counts = {split: 0 for split in SPLITS}
    source_digest = hashlib.sha256()
    files = {
        split: (staging_dir / f"{split}.documents.jsonl").open("w", encoding="utf-8", newline="\n")
        for split in SPLITS
    }
    try:
        for row in documents:
            if isinstance(row, str):
                text = row
                explicit_id = None
            elif isinstance(row, Mapping):
                text = row.get(config.data.text_field)
                explicit_id = row.get("id", row.get("document_id"))
            else:
                continue
            normalized = normalize_document(text)  # type: ignore[arg-type]
            if normalized is None:
                continue
            document_id = (
                str(explicit_id)
                if explicit_id is not None
                else hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            )
            _update_source_digest(source_digest, document_id, normalized)
            split = split_for_id(
                document_id,
                seed=config.data.seed,
                validation_fraction=config.data.validation_fraction,
            )
            json.dump(normalized, files[split], ensure_ascii=False)
            files[split].write("\n")
            counts[split] += 1
        for spool_file in files.values():
            spool_file.flush()
            os.fsync(spool_file.fileno())
    finally:
        for spool_file in files.values():
            spool_file.close()
    return counts, source_digest.hexdigest()


def _read_spool(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as spool_file:
        for line in spool_file:
            value = json.loads(line)
            if not isinstance(value, str):
                raise ValueError(f"invalid spooled document in {path}")
            yield value


def _encode_documents(texts: Iterable[str], tokenizer: Any, eos_id: int) -> Iterator[np.ndarray]:
    for text in texts:
        ids = tokenizer.encode(text).ids
        ids.append(eos_id)
        yield np.asarray(ids, dtype=np.uint16)


def _update_source_digest(digest: Any, document_id: str, text: str) -> None:
    for value in (document_id, text):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def _is_commit_sha(revision: object) -> bool:
    return (
        isinstance(revision, str)
        and len(revision) in {40, 64}
        and all(character in "0123456789abcdef" for character in revision)
    )


def _promote_transaction(promotions: list[tuple[Path, Path]], transaction_id: str) -> None:
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for _, destination in promotions:
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.{transaction_id}.bak")
                os.replace(destination, backup)
                backups[destination] = backup
        for source, destination in promotions:
            os.replace(source, destination)
            promoted.append(destination)
    except BaseException:
        for destination in reversed(promoted):
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
