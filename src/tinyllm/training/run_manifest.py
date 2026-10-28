"""Portable provenance records for training runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Mapping

from tinyllm.config.schema import TinyLLMConfig


def _git_revision() -> str | None:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _runtime_metadata() -> dict[str, str]:
    runtime = {"python": platform.python_version()}
    for distribution in ("tinyllm", "torch", "tokenizers", "datasets"):
        try:
            runtime[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        import torch
    except ImportError:
        pass
    else:
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        if cuda_version:
            runtime["cuda"] = str(cuda_version)
    return runtime


def _system_metadata() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def _accelerator_name() -> str | None:
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "Apple MPS"
    return None


def _portable_path(path: Path, repository_root: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _portable_value(value: object, repository_root: Path) -> object:
    if isinstance(value, Path):
        return _portable_path(value, repository_root)
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            return _portable_path(candidate, repository_root)
        return value
    if isinstance(value, Mapping):
        return {str(key): _portable_value(item, repository_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_value(item, repository_root) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _artifact_digests(config: TinyLLMConfig, args: argparse.Namespace) -> dict[str, str]:
    candidates: dict[str, Path] = {}
    for section_name, section in config:
        for field_name, value in section:
            if isinstance(value, Path) and value.is_file():
                candidates[f"config.{section_name}.{field_name}"] = value
    for name, value in vars(args).items():
        if isinstance(value, Path) and value.is_file():
            candidates[f"arguments.{name}"] = value
    interrupt_checkpoint = config.training.checkpoint_dir / "interrupt.pt"
    if interrupt_checkpoint.is_file():
        candidates["checkpoints.interrupt"] = interrupt_checkpoint

    digests: dict[str, str] = {}
    for name, path in sorted(candidates.items()):
        with path.open("rb") as artifact:
            digests[name] = hashlib.file_digest(artifact, "sha256").hexdigest()
    corpus_manifest = config.data.output_dir / "manifest.json"
    try:
        corpus = json.loads(corpus_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return digests
    if not isinstance(corpus, dict):
        return digests
    manifest_digest = _file_sha256(corpus_manifest)
    if manifest_digest is not None:
        digests["data.corpus_manifest"] = manifest_digest
    tokenizer_digest = corpus.get("tokenizer_sha256")
    if isinstance(tokenizer_digest, str):
        digests["data.tokenizer"] = tokenizer_digest
    corpus_artifacts = corpus.get("artifacts")
    if isinstance(corpus_artifacts, dict):
        for split, artifact in corpus_artifacts.items():
            if isinstance(split, str) and isinstance(artifact, dict):
                digest = artifact.get("sha256")
                if isinstance(digest, str) and len(digest) == 64:
                    digests[f"data.{split}_tokens"] = digest
    return digests


def start_run_manifest(
    config: TinyLLMConfig, args: argparse.Namespace, *, started_at: str
) -> tuple[Path, dict[str, object]]:
    """Start a run or continue the manifest matching a resumed checkpoint."""
    run_root = config.logging.run_dir
    manifest_path = run_root / "run.json"
    run_root.mkdir(parents=True, exist_ok=True)
    manifest_candidates = [manifest_path]
    manifest_candidates.extend(
        sorted(
            run_root.glob("run-*.json"),
            key=lambda candidate: candidate.stat().st_mtime_ns,
            reverse=True,
        )
    )
    previous: dict[str, object] | None = None
    previous_path: Path | None = None
    current_manifest: dict[str, object] | None = None
    loaded_manifests: list[tuple[Path, dict[str, object]]] = []
    for candidate in manifest_candidates:
        if not candidate.is_file():
            continue
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            loaded_manifests.append((candidate, loaded))
    if loaded_manifests:
        previous_path, previous = loaded_manifests[0]
        if previous_path == manifest_path:
            current_manifest = previous
    if manifest_path.is_file() and current_manifest is None:
        invalid_archive = run_root / f"run-invalid-{uuid.uuid4().hex}.json"
        shutil.copy2(manifest_path, invalid_archive)

    resume_path = getattr(args, "resume", None)
    resume_sha = _file_sha256(resume_path) if isinstance(resume_path, Path) else None
    matching_manifest: tuple[Path, dict[str, object]] | None = None
    if resume_sha is not None and isinstance(resume_path, Path):
        for candidate_path, candidate_payload in loaded_manifests:
            artifacts = candidate_payload.get("artifacts")
            has_matching_artifact = isinstance(artifacts, dict) and resume_sha in artifacts.values()
            if has_matching_artifact:
                matching_manifest = (candidate_path, candidate_payload)
                break

    if matching_manifest is not None:
        previous_path, previous = matching_manifest
        if previous_path != manifest_path and manifest_path.is_file():
            if current_manifest is not None:
                _archive_current_manifest(manifest_path, current_manifest)
        payload = previous
        history = payload.setdefault("resume_history", [])
        if not isinstance(history, list):
            history = []
            payload["resume_history"] = history
        history.append(
            {
                "started_at": started_at,
                "checkpoint_sha256": resume_sha,
                "arguments": _portable_value(
                    {
                        name: value
                        for name, value in vars(args).items()
                        if name not in {"handler", "func"}
                    },
                    Path(__file__).resolve().parents[3],
                ),
            }
        )
        payload["status"] = "running"
        payload["finished_at"] = None
    else:
        if current_manifest is not None:
            _archive_current_manifest(manifest_path, current_manifest)
        payload = build_run_manifest(
            config,
            args,
            status="running",
            started_at=started_at,
            finished_at=None,
        )
    payload["status"] = "running"
    payload["finished_at"] = None
    write_run_manifest(manifest_path, payload)
    return manifest_path, payload


def _archive_current_manifest(path: Path, payload: Mapping[str, object]) -> None:
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not path.is_file():
        return
    if len(run_id) != 32 or any(char not in "0123456789abcdef" for char in run_id):
        run_id = uuid.uuid4().hex
    archive_path = path.parent / f"run-{run_id}.json"
    if archive_path.exists():
        archive_path = path.parent / f"run-{run_id}-{uuid.uuid4().hex[:8]}.json"
    shutil.copy2(path, archive_path)


def _file_sha256(path: Path) -> str | None:
    try:
        with path.open("rb") as artifact:
            return hashlib.file_digest(artifact, "sha256").hexdigest()
    except OSError:
        return None


def finish_run_manifest(
    path: Path,
    payload: dict[str, object],
    config: TinyLLMConfig,
    args: argparse.Namespace,
    *,
    status: str,
    finished_at: str,
) -> None:
    """Finalize status and refresh output artifact digests for later resume matching."""
    payload["status"] = status
    payload["finished_at"] = finished_at
    payload["artifacts"] = _artifact_digests(config, args)
    write_run_manifest(path, payload)


def build_run_manifest(
    config: TinyLLMConfig,
    args: argparse.Namespace,
    *,
    status: str,
    started_at: str,
    finished_at: str | None,
) -> dict[str, object]:
    """Build a versioned run record without machine-private paths or variables."""
    if status not in {"running", "completed", "failed", "interrupted"}:
        raise ValueError(f"unsupported run status: {status}")
    repository_root = Path(__file__).resolve().parents[3]
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": uuid.uuid4().hex,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "config": _portable_value(config.model_dump(mode="python"), repository_root),
        "arguments": _portable_value(
            {
                name: value
                for name, value in vars(args).items()
                if name not in {"handler", "func"}
            },
            repository_root,
        ),
        "seeds": {
            "data": config.data.seed,
            "training": config.training.seed,
            "generation": config.generation.seed,
        },
        "artifacts": _artifact_digests(config, args),
    }
    revision = _git_revision()
    if revision is not None:
        payload["source_revision"] = revision
    runtime = _runtime_metadata()
    if runtime:
        payload["runtime"] = runtime
    system = _system_metadata()
    if system:
        payload["system"] = system
    accelerator = _accelerator_name()
    if accelerator is not None:
        payload["accelerator"] = accelerator
    return payload


def write_run_manifest(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace a UTF-8 JSON manifest, preserving the prior file on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, sort_keys=True, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

