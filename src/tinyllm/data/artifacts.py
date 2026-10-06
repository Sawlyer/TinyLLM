"""Checksummed, atomic corpus artifact metadata and I/O."""

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tokenizers import Tokenizer

UINT16_DTYPE = np.dtype("<u2")
LEGACY_MANIFEST_VERSION = 1
MANIFEST_VERSION = 2
REQUIRED_SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>", "<pad>")


@dataclass(frozen=True)
class TokenArtifactManifest:
    split: str
    filename: str
    dtype: str
    token_count: int
    tokenizer_sha256: str
    sha256: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.split, "split")
        _require_non_empty_string(self.filename, "filename")
        if self.dtype != "uint16":
            raise ValueError(f"unsupported token artifact dtype: {self.dtype}")
        _require_integer(self.token_count, "token_count", minimum=0)
        _require_sha256(self.tokenizer_sha256, "tokenizer_sha256")
        _require_sha256(self.sha256, "sha256")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TokenArtifactManifest":
        required = {"split", "filename", "dtype", "token_count", "tokenizer_sha256", "sha256"}
        if set(raw) != required:
            raise ValueError("invalid token artifact manifest fields")
        return cls(
            split=_require_non_empty_string(raw["split"], "split"),
            filename=_require_non_empty_string(raw["filename"], "filename"),
            dtype=_require_non_empty_string(raw["dtype"], "dtype"),
            token_count=_require_integer(raw["token_count"], "token_count", minimum=0),
            tokenizer_sha256=_require_sha256(raw["tokenizer_sha256"], "tokenizer_sha256"),
            sha256=_require_sha256(raw["sha256"], "sha256"),
        )


@dataclass(frozen=True)
class CorpusManifest:
    version: int
    dataset_name: str
    dataset_requested_revision: str | None
    dataset_revision: str
    dataset_fingerprint: str
    text_field: str
    seed: int
    validation_fraction: float
    tokenizer_path: str
    tokenizer_sha256: str
    tokenizer_config: dict[str, Any]
    data_config: dict[str, Any]
    document_counts: dict[str, int]
    artifacts: dict[str, TokenArtifactManifest]
    tokenizer_vocab_size: int | None = None
    special_token_ids: dict[str, int] | None = None

    def __post_init__(self) -> None:
        version = _require_integer(self.version, "version", minimum=1)
        if version not in {LEGACY_MANIFEST_VERSION, MANIFEST_VERSION}:
            raise ValueError(f"unsupported corpus manifest version: {version}")
        _require_non_empty_string(self.dataset_name, "dataset_name")
        if self.dataset_requested_revision is not None:
            _require_non_empty_string(
                self.dataset_requested_revision, "dataset_requested_revision"
            )
        _require_immutable_revision(self.dataset_revision)
        _require_non_empty_string(self.dataset_fingerprint, "dataset_fingerprint")
        _require_non_empty_string(self.text_field, "text_field")
        _require_integer(self.seed, "seed")
        _require_fraction(self.validation_fraction)
        _require_non_empty_string(self.tokenizer_path, "tokenizer_path")
        _require_sha256(self.tokenizer_sha256, "tokenizer_sha256")
        if not isinstance(self.tokenizer_config, dict) or not self.tokenizer_config:
            raise ValueError("tokenizer_config must be a non-empty mapping")
        if not isinstance(self.data_config, dict) or not self.data_config:
            raise ValueError("data_config must be a non-empty mapping")
        if set(self.document_counts) != {"train", "validation"}:
            raise ValueError("document_counts must contain train and validation")
        for split, count in self.document_counts.items():
            _require_integer(count, f"document_counts.{split}", minimum=0)
        if set(self.artifacts) != {"train", "validation"}:
            raise ValueError("manifest must contain train and validation artifacts")
        for split, artifact in self.artifacts.items():
            if not isinstance(artifact, TokenArtifactManifest):
                raise ValueError(f"artifact {split} has invalid metadata")
            if artifact.split != split:
                raise ValueError(f"artifact split mismatch for {split}")
            if artifact.tokenizer_sha256 != self.tokenizer_sha256:
                raise ValueError(f"artifact tokenizer checksum mismatch for {split}")

        if version == MANIFEST_VERSION:
            _require_integer(
                self.tokenizer_vocab_size,
                "tokenizer_vocab_size",
                minimum=1,
            )
            special_token_ids = self.special_token_ids
            if not isinstance(special_token_ids, dict):
                raise ValueError("special_token_ids must be a mapping")
            if set(special_token_ids) != set(REQUIRED_SPECIAL_TOKENS):
                raise ValueError("special_token_ids must contain required tokenizer tokens")
            validated_ids = [
                _require_integer(token_id, f"special_token_ids.{token}", minimum=0)
                for token, token_id in special_token_ids.items()
            ]
            if len(set(validated_ids)) != len(validated_ids):
                raise ValueError("special_token_ids must be distinct")
            if any(token_id >= self.tokenizer_vocab_size for token_id in validated_ids):
                raise ValueError("special_token_ids exceed tokenizer_vocab_size")
        elif self.tokenizer_vocab_size is not None or self.special_token_ids not in (None, {}):
            raise ValueError("legacy corpus manifest cannot contain version 2 tokenizer identity")

        expected_data_identity = {
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_requested_revision,
            "text_field": self.text_field,
            "seed": self.seed,
            "validation_fraction": self.validation_fraction,
        }
        for field, expected in expected_data_identity.items():
            if self.data_config.get(field) != expected:
                raise ValueError(f"data_config {field} is inconsistent with manifest")

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        if self.version == LEGACY_MANIFEST_VERSION:
            raw.pop("tokenizer_vocab_size")
            raw.pop("special_token_ids")
        raw["artifacts"] = {
            split: asdict(artifact) for split, artifact in self.artifacts.items()
        }
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CorpusManifest":
        base_required = {
            "version",
            "dataset_name",
            "dataset_requested_revision",
            "dataset_revision",
            "dataset_fingerprint",
            "text_field",
            "seed",
            "validation_fraction",
            "tokenizer_path",
            "tokenizer_sha256",
            "tokenizer_config",
            "data_config",
            "document_counts",
            "artifacts",
        }
        version = _require_integer(raw.get("version"), "version", minimum=1)
        if version == MANIFEST_VERSION:
            required = base_required | {"tokenizer_vocab_size", "special_token_ids"}
        elif version == LEGACY_MANIFEST_VERSION:
            required = base_required
        else:
            raise ValueError(f"unsupported corpus manifest version: {version}")
        if set(raw) != required:
            raise ValueError("invalid corpus manifest fields")
        artifacts_raw = raw["artifacts"]
        if not isinstance(artifacts_raw, Mapping):
            raise ValueError("manifest artifacts must be a mapping")
        artifacts = {
            str(split): TokenArtifactManifest.from_dict(value)
            for split, value in artifacts_raw.items()
            if isinstance(value, Mapping)
        }
        if len(artifacts) != len(artifacts_raw) or set(artifacts) != {"train", "validation"}:
            raise ValueError("manifest must contain train and validation artifacts")
        counts_raw = raw["document_counts"]
        if not isinstance(counts_raw, Mapping):
            raise ValueError("document_counts must be a mapping")
        document_counts = {
            _require_non_empty_string(split, "document_counts split"): _require_integer(
                count, f"document_counts.{split}", minimum=0
            )
            for split, count in counts_raw.items()
        }
        if set(document_counts) != {"train", "validation"}:
            raise ValueError("document_counts must contain train and validation")
        tokenizer_config = raw["tokenizer_config"]
        data_config = raw["data_config"]
        if not isinstance(tokenizer_config, Mapping) or not isinstance(data_config, Mapping):
            raise ValueError("manifest configuration snapshots must be mappings")
        raw_special_ids = raw.get("special_token_ids")
        if raw_special_ids is not None and not isinstance(raw_special_ids, Mapping):
            raise ValueError("special_token_ids must be a mapping")
        return cls(
            version=version,
            dataset_name=_require_non_empty_string(raw["dataset_name"], "dataset_name"),
            dataset_requested_revision=(
                None
                if raw["dataset_requested_revision"] is None
                else _require_non_empty_string(
                    raw["dataset_requested_revision"], "dataset_requested_revision"
                )
            ),
            dataset_revision=_require_immutable_revision(raw["dataset_revision"]),
            dataset_fingerprint=_require_non_empty_string(
                raw["dataset_fingerprint"], "dataset_fingerprint"
            ),
            text_field=_require_non_empty_string(raw["text_field"], "text_field"),
            seed=_require_integer(raw["seed"], "seed"),
            validation_fraction=_require_fraction(raw["validation_fraction"]),
            tokenizer_path=_require_non_empty_string(raw["tokenizer_path"], "tokenizer_path"),
            tokenizer_sha256=_require_sha256(raw["tokenizer_sha256"], "tokenizer_sha256"),
            tokenizer_config=dict(tokenizer_config),
            data_config=dict(data_config),
            document_counts=document_counts,
            artifacts=artifacts,
            tokenizer_vocab_size=(
                None
                if version == LEGACY_MANIFEST_VERSION
                else _require_integer(
                    raw["tokenizer_vocab_size"],
                    "tokenizer_vocab_size",
                    minimum=1,
                )
            ),
            special_token_ids=(
                None
                if version == LEGACY_MANIFEST_VERSION
                else {
                    _require_non_empty_string(token, "special token"): _require_integer(
                        token_id,
                        f"special_token_ids.{token}",
                        minimum=0,
                    )
                    for token, token_id in raw_special_ids.items()
                }
            ),
        )


def _require_non_empty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _require_fraction(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("validation_fraction must be a number between 0 and 1")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    return fraction


def _require_sha256(value: object, field: str) -> str:
    digest = _require_non_empty_string(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _require_immutable_revision(value: object) -> str:
    revision = _require_non_empty_string(value, "dataset_revision")
    if revision.startswith("injected-sha256:"):
        _require_sha256(revision.removeprefix("injected-sha256:"), "dataset_revision")
        return revision
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("dataset_revision must be an immutable commit SHA")
    return revision


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_token_artifact(
    output_dir: Path,
    split: str,
    tokens: NDArray[np.uint16],
    *,
    tokenizer_sha256: str,
    overwrite: bool = False,
) -> TokenArtifactManifest:
    """Atomically write one uint16 token array."""
    array = np.asarray(tokens)
    if array.dtype != np.uint16:
        raise TypeError("token artifacts require uint16 arrays")
    return write_token_chunks(
        output_dir,
        split,
        [array],
        tokenizer_sha256=tokenizer_sha256,
        overwrite=overwrite,
    )


def write_token_chunks(
    output_dir: Path,
    split: str,
    chunks: Iterable[NDArray[np.uint16]],
    *,
    tokenizer_sha256: str,
    overwrite: bool = False,
) -> TokenArtifactManifest:
    """Stream uint16 chunks into one atomic binary artifact."""
    split = _require_non_empty_string(split, "split")
    tokenizer_sha256 = _require_sha256(tokenizer_sha256, "tokenizer_sha256")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{split}.bin"
    destination = output_dir / filename
    temporary = destination.with_name(f"{destination.name}.tmp")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"token artifact already exists: {destination}")

    token_count = 0
    digest = hashlib.sha256()
    try:
        temporary.unlink(missing_ok=True)
        with temporary.open("wb") as artifact_file:
            for chunk in chunks:
                array = np.asarray(chunk)
                if array.dtype != np.uint16:
                    raise TypeError("token artifacts require uint16 arrays")
                encoded = np.ascontiguousarray(array, dtype=UINT16_DTYPE).tobytes()
                artifact_file.write(encoded)
                digest.update(encoded)
                token_count += array.size
            artifact_file.flush()
            os.fsync(artifact_file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return TokenArtifactManifest(
        split=split,
        filename=filename,
        dtype="uint16",
        token_count=token_count,
        tokenizer_sha256=tokenizer_sha256,
        sha256=digest.hexdigest(),
    )


def open_token_artifact(
    output_dir: Path, manifest: TokenArtifactManifest
) -> np.memmap | NDArray[np.uint16]:
    """Validate checksum and shape before opening a read-only memory map."""
    if manifest.dtype != "uint16":
        raise ValueError(f"unsupported token artifact dtype: {manifest.dtype}")
    output_dir = Path(output_dir).resolve()
    artifact_path = (output_dir / manifest.filename).resolve()
    if artifact_path.parent != output_dir:
        raise ValueError("token artifact path escapes output directory")
    if not artifact_path.is_file():
        raise ValueError(f"token artifact missing: {artifact_path}")
    expected_size = manifest.token_count * UINT16_DTYPE.itemsize
    if artifact_path.stat().st_size != expected_size:
        raise ValueError(f"token artifact size mismatch: {artifact_path}")
    if sha256_file(artifact_path) != manifest.sha256:
        raise ValueError(f"token artifact checksum mismatch: {artifact_path}")
    if manifest.token_count == 0:
        return np.array([], dtype=np.uint16)
    return np.memmap(artifact_path, dtype=UINT16_DTYPE, mode="r", shape=(manifest.token_count,))


def write_manifest(path: Path, manifest: CorpusManifest, *, overwrite: bool = False) -> None:
    """Atomically write deterministic JSON manifest metadata."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"corpus manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        with temporary.open("w", encoding="utf-8", newline="\n") as manifest_file:
            json.dump(manifest.to_dict(), manifest_file, indent=2, sort_keys=True)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(
    path: Path,
    *,
    splits: Iterable[str] | None = None,
) -> CorpusManifest:
    """Load and fully validate a corpus manifest and referenced files."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid corpus manifest: {path}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("corpus manifest root must be a mapping")
    manifest = CorpusManifest.from_dict(raw)

    tokenizer_path = Path(manifest.tokenizer_path)
    if not tokenizer_path.is_absolute():
        tokenizer_path = path.parent / tokenizer_path
    if not tokenizer_path.is_file():
        raise ValueError(f"tokenizer artifact missing: {tokenizer_path}")
    if sha256_file(tokenizer_path) != manifest.tokenizer_sha256:
        raise ValueError(f"tokenizer checksum mismatch: {tokenizer_path}")
    actual_vocab_size, actual_special_ids = _read_tokenizer_identity(tokenizer_path)
    if manifest.version == LEGACY_MANIFEST_VERSION:
        manifest = replace(
            manifest,
            version=MANIFEST_VERSION,
            tokenizer_vocab_size=actual_vocab_size,
            special_token_ids=actual_special_ids,
        )
    elif (
        manifest.tokenizer_vocab_size != actual_vocab_size
        or manifest.special_token_ids != actual_special_ids
    ):
        raise ValueError("tokenizer vocabulary identity differs from corpus manifest")

    selected_splits = set(manifest.artifacts) if splits is None else set(splits)
    unknown_splits = selected_splits - set(manifest.artifacts)
    if unknown_splits:
        raise ValueError(f"unknown manifest splits: {sorted(unknown_splits)}")
    for split in selected_splits:
        artifact = manifest.artifacts[split]
        if artifact.split != split:
            raise ValueError(f"artifact split mismatch for {split}")
        if artifact.tokenizer_sha256 != manifest.tokenizer_sha256:
            raise ValueError(f"artifact tokenizer checksum mismatch for {split}")
        open_token_artifact(path.parent, artifact)
    object.__setattr__(manifest, "_manifest_path", path.resolve())
    return manifest


def _read_tokenizer_identity(path: Path) -> tuple[int, dict[str, int]]:
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as error:
        raise ValueError(f"invalid tokenizer artifact: {path}") from error
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("tokenizer vocabulary must be non-empty")
    special_ids: dict[str, int] = {}
    for token in REQUIRED_SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token {token}")
        special_ids[token] = token_id
    return vocab_size, special_ids
