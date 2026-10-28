import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tinyllm.cli import main
from tinyllm.config.load import load_config
from tinyllm.config.schema import TinyLLMConfig
from tinyllm.data.audit import audit_artifacts
from tinyllm.data.prepare import prepare_corpus


@pytest.fixture
def prepared_corpus(tmp_path: Path) -> tuple[Path, Path]:
    root = Path(__file__).parents[2]
    raw = load_config(root / "configs" / "tinyllm.yaml", []).model_dump(mode="python")
    raw["data"].update(
        {
            "cache_dir": tmp_path / "cache",
            "output_dir": tmp_path / "processed",
            "validation_fraction": 0.5,
        }
    )
    raw["tokenizer"].update(
        {"vocab_size": 128, "min_frequency": 1, "path": tmp_path / "processed" / "tokenizer.json"}
    )
    raw["model"]["vocab_size"] = 128
    config = TinyLLMConfig.model_validate(raw)
    documents = [
        {"id": f"doc-{index}", "text": f"sample text number {index}"} for index in range(24)
    ]
    prepare_corpus(config, documents=documents)
    manifest_path = config.data.output_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["tokenizer_path"] = "tokenizer.json"
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return manifest_path, config.data.output_dir


def test_audit_reports_valid_artifacts_and_is_read_only(
    prepared_corpus: tuple[Path, Path],
) -> None:
    manifest_path, output_dir = prepared_corpus
    before = {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()}

    report = audit_artifacts(manifest_path)

    after = {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()}
    assert report["status"] == "ok"
    assert report["errors"] == []
    assert report["warnings"] == []
    assert {item["split"] for item in report["artifacts"]} == {"train", "validation"}
    assert before == after


def test_audit_reports_truncated_artifact_and_does_not_modify_it(
    prepared_corpus: tuple[Path, Path],
) -> None:
    manifest_path, output_dir = prepared_corpus
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = output_dir / raw["artifacts"]["train"]["filename"]
    artifact_path.write_bytes(artifact_path.read_bytes()[:-1])
    truncated = artifact_path.read_bytes()

    report = audit_artifacts(manifest_path)

    assert report["status"] == "error"
    assert any("size" in error["message"] for error in report["errors"])
    assert artifact_path.read_bytes() == truncated


def test_audit_detects_same_size_checksum_mismatch(
    prepared_corpus: tuple[Path, Path],
) -> None:
    manifest_path, output_dir = prepared_corpus
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = output_dir / raw["artifacts"]["train"]["filename"]
    original = artifact_path.read_bytes()
    artifact_path.write_bytes(bytes([original[0] ^ 1]) + original[1:])

    report = audit_artifacts(manifest_path)

    assert report["status"] == "error"
    assert any("checksum" in error["message"] for error in report["errors"])


def test_audit_detects_tokenizer_substitution(
    prepared_corpus: tuple[Path, Path],
) -> None:
    manifest_path, output_dir = prepared_corpus
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")

    report = audit_artifacts(manifest_path)

    assert report["status"] == "error"
    assert any("tokenizer" in error["message"].lower() for error in report["errors"])


def test_audit_cli_prints_machine_readable_success_report(
    prepared_corpus: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, _ = prepared_corpus

    main(["audit-artifacts", "--manifest", str(manifest_path)])

    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "ok"
    assert captured.err == ""


def test_audit_cli_returns_nonzero_after_printing_error_report(
    prepared_corpus: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path, output_dir = prepared_corpus
    artifact_path = output_dir / "train.bin"
    artifact_path.write_bytes(artifact_path.read_bytes()[:-1])

    with pytest.raises(SystemExit) as error:
        main(["audit-artifacts", "--manifest", str(manifest_path)])

    captured = capsys.readouterr()
    assert error.value.code == 1
    assert json.loads(captured.out)["status"] == "error"
    assert "artifact audit found errors" in captured.err


def test_audit_warns_for_legacy_manifest_without_upgrading_on_disk(
    prepared_corpus: tuple[Path, Path],
) -> None:
    manifest_path, _ = prepared_corpus
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["version"] = 1
    raw.pop("tokenizer_vocab_size")
    raw.pop("special_token_ids")
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    legacy_bytes = manifest_path.read_bytes()

    report = audit_artifacts(manifest_path)

    assert report["status"] == "warning"
    assert any("legacy" in warning["message"].lower() for warning in report["warnings"])
    assert manifest_path.read_bytes() == legacy_bytes


def test_audit_resolves_relative_files_after_dataset_directory_move(
    prepared_corpus: tuple[Path, Path], tmp_path: Path
) -> None:
    manifest_path, output_dir = prepared_corpus
    moved_dir = tmp_path / "moved" / "dataset"
    moved_dir.parent.mkdir()
    shutil.move(str(output_dir), moved_dir)

    report = audit_artifacts(moved_dir / "manifest.json")

    assert report["status"] == "ok"
    assert all(Path(item["path"]).parent == moved_dir.resolve() for item in report["artifacts"])


def test_audit_detects_out_of_range_token_ids(
    prepared_corpus: tuple[Path, Path],
) -> None:
    manifest_path, output_dir = prepared_corpus
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_meta = raw["artifacts"]["train"]
    artifact_path = output_dir / artifact_meta["filename"]
    token_bytes = bytearray(artifact_path.read_bytes())
    vocab_size = raw["tokenizer_vocab_size"]
    token_bytes[0:2] = vocab_size.to_bytes(2, "little")
    artifact_path.write_bytes(token_bytes)
    artifact_meta["sha256"] = hashlib.sha256(token_bytes).hexdigest()
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    report = audit_artifacts(manifest_path)

    assert report["status"] == "error"
    assert any("vocabulary" in error["message"].lower() for error in report["errors"])
