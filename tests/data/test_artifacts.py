from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tinyllm.data.artifacts import (
    CorpusManifest,
    TokenArtifactManifest,
    open_token_artifact,
    write_token_artifact,
)

TOKENIZER_SHA256 = "a" * 64
ARTIFACT_SHA256 = "b" * 64


def test_binary_artifact_round_trip(tmp_path: Path) -> None:
    tokens = np.array([1, 8, 3, 2], dtype=np.uint16)

    manifest = write_token_artifact(
        tmp_path, "train", tokens, tokenizer_sha256=TOKENIZER_SHA256
    )
    loaded = open_token_artifact(tmp_path, manifest)

    assert loaded.tolist() == tokens.tolist()
    assert manifest.dtype == "uint16"
    assert manifest.token_count == 4
    assert not (tmp_path / "train.bin.tmp").exists()


def test_binary_artifact_rejects_corruption(tmp_path: Path) -> None:
    tokens = np.array([1, 8, 3, 2], dtype=np.uint16)
    manifest = write_token_artifact(
        tmp_path, "train", tokens, tokenizer_sha256=TOKENIZER_SHA256
    )
    artifact_path = tmp_path / manifest.filename
    artifact_path.write_bytes(artifact_path.read_bytes()[:-1] + b"\xff")

    with pytest.raises(ValueError, match="checksum"):
        open_token_artifact(tmp_path, manifest)


def test_binary_artifact_refuses_overwrite_by_default(tmp_path: Path) -> None:
    original = np.array([1, 2], dtype=np.uint16)
    replacement = np.array([9, 9], dtype=np.uint16)
    manifest = write_token_artifact(
        tmp_path, "train", original, tokenizer_sha256=TOKENIZER_SHA256
    )

    with pytest.raises(FileExistsError, match="train.bin"):
        write_token_artifact(
            tmp_path, "train", replacement, tokenizer_sha256=TOKENIZER_SHA256
        )

    assert open_token_artifact(tmp_path, manifest).tolist() == original.tolist()


def test_binary_artifact_requires_uint16(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="uint16"):
        write_token_artifact(
            tmp_path,
            "train",
            np.array([1, 2], dtype=np.int64),
            tokenizer_sha256=TOKENIZER_SHA256,
        )


def test_binary_artifact_rejects_non_uint16_manifest(tmp_path: Path) -> None:
    tokens = np.array([1, 2], dtype=np.uint16)
    manifest = write_token_artifact(
        tmp_path, "train", tokens, tokenizer_sha256=TOKENIZER_SHA256
    )

    with pytest.raises(ValueError, match="dtype"):
        open_token_artifact(tmp_path, replace(manifest, dtype="uint32"))


@pytest.mark.parametrize(
    ("split", "tokenizer_sha256", "field"),
    [("", TOKENIZER_SHA256, "split"), ("train", "", "tokenizer_sha256")],
)
def test_binary_artifact_rejects_invalid_identity_before_writing(
    tmp_path: Path, split: str, tokenizer_sha256: str, field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        write_token_artifact(
            tmp_path,
            split,
            np.array([1, 2], dtype=np.uint16),
            tokenizer_sha256=tokenizer_sha256,
        )

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("token_count", 1.9), ("token_count", True)],
)
def test_token_artifact_manifest_rejects_non_integer_count(field: str, value: object) -> None:
    raw = _token_artifact_dict("train")
    raw[field] = value

    with pytest.raises(ValueError, match="token_count.*integer"):
        TokenArtifactManifest.from_dict(raw)


@pytest.mark.parametrize("field", ["split", "filename", "tokenizer_sha256", "sha256"])
def test_token_artifact_manifest_rejects_empty_identity(field: str) -> None:
    raw = _token_artifact_dict("train")
    raw[field] = ""

    with pytest.raises(ValueError, match=field):
        TokenArtifactManifest.from_dict(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", 1.9), ("seed", 7.5), ("seed", True)],
)
def test_corpus_manifest_rejects_lossy_integer_fields(field: str, value: object) -> None:
    raw = _corpus_manifest_dict()
    raw[field] = value

    with pytest.raises(ValueError, match=f"{field}.*integer"):
        CorpusManifest.from_dict(raw)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_corpus_manifest_rejects_invalid_validation_fraction(fraction: float) -> None:
    raw = _corpus_manifest_dict()
    raw["validation_fraction"] = fraction

    with pytest.raises(ValueError, match="validation_fraction.*between 0 and 1"):
        CorpusManifest.from_dict(raw)


def test_corpus_manifest_rejects_incoherent_tokenizer_identity() -> None:
    raw = _corpus_manifest_dict()
    raw["artifacts"]["train"]["tokenizer_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="tokenizer checksum mismatch"):
        CorpusManifest.from_dict(raw)


def test_corpus_manifest_rejects_empty_dataset_name() -> None:
    raw = _corpus_manifest_dict()
    raw["dataset_name"] = ""

    with pytest.raises(ValueError, match="dataset_name"):
        CorpusManifest.from_dict(raw)


def test_legacy_manifest_round_trip_omits_version_two_identity_fields() -> None:
    raw = _corpus_manifest_dict()

    assert CorpusManifest.from_dict(raw).to_dict() == raw


def _token_artifact_dict(split: str) -> dict[str, object]:
    return {
        "split": split,
        "filename": f"{split}.bin",
        "dtype": "uint16",
        "token_count": 2,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "sha256": ARTIFACT_SHA256,
    }


def _corpus_manifest_dict() -> dict[str, object]:
    return {
        "version": 1,
        "dataset_name": "example/dataset",
        "dataset_requested_revision": "main",
        "dataset_revision": "d" * 40,
        "dataset_fingerprint": "dataset-fingerprint-123",
        "text_field": "text",
        "seed": 7,
        "validation_fraction": 0.25,
        "tokenizer_path": "tokenizer.json",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_config": {
            "vocab_size": 128,
            "min_frequency": 1,
            "special_tokens": ["<unk>", "<bos>", "<eos>", "<pad>"],
            "path": "tokenizer.json",
        },
        "data_config": {
            "dataset_name": "example/dataset",
            "dataset_revision": "main",
            "cache_dir": "cache",
            "output_dir": "processed",
            "text_field": "text",
            "validation_fraction": 0.25,
            "seed": 7,
        },
        "document_counts": {"train": 1, "validation": 1},
        "artifacts": {
            "train": _token_artifact_dict("train"),
            "validation": _token_artifact_dict("validation"),
        },
    }
