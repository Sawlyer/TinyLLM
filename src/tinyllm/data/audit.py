"""Read-only checks for prepared corpus artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tinyllm.data.artifacts import (
    LEGACY_MANIFEST_VERSION,
    CorpusManifest,
    _read_tokenizer_identity,
    sha256_file,
)


def audit_artifacts(manifest_path: Path) -> dict[str, Any]:
    """Audit a corpus manifest and its referenced files without changing them."""
    manifest_path = Path(manifest_path).resolve()
    report: dict[str, Any] = {
        "status": "ok",
        "manifest": str(manifest_path),
        "version": None,
        "tokenizer": None,
        "artifacts": [],
        "warnings": [],
        "errors": [],
    }

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _add_error(report, "manifest", f"cannot read manifest: {error}")
        return _finish(report)
    if not isinstance(raw, Mapping):
        _add_error(report, "manifest", "manifest root must be a JSON object")
        return _finish(report)

    try:
        manifest = CorpusManifest.from_dict(raw)
    except (TypeError, ValueError) as error:
        _add_error(report, "manifest", str(error))
        return _finish(report)

    report["version"] = manifest.version
    if manifest.version == LEGACY_MANIFEST_VERSION:
        _add_warning(
            report,
            "manifest",
            "legacy manifest has no recorded tokenizer vocabulary identity",
        )

    tokenizer_path = Path(manifest.tokenizer_path)
    if not tokenizer_path.is_absolute():
        tokenizer_path = manifest_path.parent / tokenizer_path
    tokenizer_path = tokenizer_path.resolve()
    tokenizer_report: dict[str, Any] = {
        "path": str(tokenizer_path),
        "expected_sha256": manifest.tokenizer_sha256,
        "actual_sha256": None,
        "size_bytes": None,
        "vocab_size": None,
        "special_token_ids": None,
    }
    report["tokenizer"] = tokenizer_report
    tokenizer_identity: tuple[int, dict[str, int]] | None = None
    if not tokenizer_path.is_file():
        _add_error(report, "tokenizer", f"tokenizer artifact missing: {tokenizer_path}")
    else:
        tokenizer_report["size_bytes"] = tokenizer_path.stat().st_size
        try:
            actual_sha256 = sha256_file(tokenizer_path)
            tokenizer_report["actual_sha256"] = actual_sha256
            if actual_sha256 != manifest.tokenizer_sha256:
                _add_error(report, "tokenizer", "tokenizer checksum mismatch")
            tokenizer_identity = _read_tokenizer_identity(tokenizer_path)
            actual_vocab_size, actual_special_ids = tokenizer_identity
            tokenizer_report["vocab_size"] = actual_vocab_size
            tokenizer_report["special_token_ids"] = actual_special_ids
            if manifest.version != LEGACY_MANIFEST_VERSION and (
                manifest.tokenizer_vocab_size != actual_vocab_size
                or manifest.special_token_ids != actual_special_ids
            ):
                _add_error(
                    report,
                    "tokenizer",
                    "tokenizer vocabulary identity differs from corpus manifest",
                )
        except (OSError, ValueError) as error:
            _add_error(report, "tokenizer", str(error))

    vocab_size = (
        manifest.tokenizer_vocab_size
        if manifest.tokenizer_vocab_size is not None
        else tokenizer_identity[0]
        if tokenizer_identity is not None
        else None
    )
    for split, artifact in sorted(manifest.artifacts.items()):
        artifact_path = (manifest_path.parent / artifact.filename).resolve()
        expected_size = artifact.token_count * np.dtype("<u2").itemsize
        item: dict[str, Any] = {
            "split": split,
            "path": str(artifact_path),
            "dtype": artifact.dtype,
            "token_count": artifact.token_count,
            "expected_size_bytes": expected_size,
            "size_bytes": None,
            "expected_sha256": artifact.sha256,
            "actual_sha256": None,
            "max_token_id": None,
        }
        report["artifacts"].append(item)

        if artifact_path.parent != manifest_path.parent:
            _add_error(report, split, "token artifact path escapes manifest directory")
            continue
        if not artifact_path.is_file():
            _add_error(report, split, f"token artifact missing: {artifact_path}")
            continue
        try:
            actual_size = artifact_path.stat().st_size
            item["size_bytes"] = actual_size
            if actual_size != expected_size:
                _add_error(
                    report,
                    split,
                    f"token artifact size mismatch: expected {expected_size}, got {actual_size}",
                )
                continue
            actual_sha256 = sha256_file(artifact_path)
            item["actual_sha256"] = actual_sha256
            if actual_sha256 != artifact.sha256:
                _add_error(report, split, "token artifact checksum mismatch")
            if artifact.token_count:
                tokens = np.memmap(
                    artifact_path,
                    dtype=np.dtype("<u2"),
                    mode="r",
                    shape=(artifact.token_count,),
                )
                maximum = int(tokens.max())
                item["max_token_id"] = maximum
                del tokens
                if vocab_size is not None and maximum >= vocab_size:
                    _add_error(
                        report,
                        split,
                        f"token ID {maximum} exceeds tokenizer vocabulary size {vocab_size}",
                    )
        except OSError as error:
            _add_error(report, split, f"cannot inspect token artifact: {error}")

        if artifact.tokenizer_sha256 != manifest.tokenizer_sha256:
            _add_error(report, split, "artifact tokenizer identity differs from manifest")

    return _finish(report)


def _add_error(report: dict[str, Any], target: str, message: str) -> None:
    report["errors"].append({"target": target, "message": message})


def _add_warning(report: dict[str, Any], target: str, message: str) -> None:
    report["warnings"].append({"target": target, "message": message})


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    if report["errors"]:
        report["status"] = "error"
    elif report["warnings"]:
        report["status"] = "warning"
    return report
