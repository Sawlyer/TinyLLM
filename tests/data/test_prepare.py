from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer

from tinyllm.config.load import load_config
from tinyllm.config.schema import TinyLLMConfig, TokenizerConfig
from tinyllm.data.artifacts import load_manifest, open_token_artifact
from tinyllm.data.clean import normalize_document
from tinyllm.data.prepare import prepare_corpus
from tinyllm.data.split import split_for_id
from tinyllm.data.tokenizer import train_tokenizer


def test_cleaning_normalizes_nfc_and_drops_blank() -> None:
    assert normalize_document("  cafe\u0301  ") == "caf\u00e9"
    assert normalize_document(" \n\t ") is None


def test_split_is_stable() -> None:
    first = split_for_id("doc-42", seed=1337, validation_fraction=0.01)

    assert first == split_for_id("doc-42", seed=1337, validation_fraction=0.01)
    assert first in {"train", "validation"}


def test_split_rejects_invalid_validation_fraction() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_for_id("doc-42", seed=1337, validation_fraction=0.0)


def test_tokenizer_training_is_atomic_and_has_required_tokens(tmp_path: Path) -> None:
    output = tmp_path / "tokenizer.json"
    config = TokenizerConfig(
        vocab_size=64,
        min_frequency=1,
        special_tokens=["<unk>", "<bos>", "<eos>", "<pad>"],
        path=output,
    )

    tokenizer = train_tokenizer(["alpha beta", "beta gamma"], config, output)
    loaded = Tokenizer.from_file(str(output))

    assert tokenizer.get_vocab() == loaded.get_vocab()
    assert all(loaded.token_to_id(token) is not None for token in config.special_tokens)
    assert not output.with_name("tokenizer.json.tmp").exists()


def test_prepare_corpus_injected_documents_adds_document_eos(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    documents = [
        {"id": f"doc-{index}", "text": f"sample document number {index}"}
        for index in range(24)
    ]
    documents.append({"id": "blank", "text": " \n "})

    manifest = prepare_corpus(config, documents=documents)
    loaded_manifest = load_manifest(config.data.output_dir / "manifest.json")
    tokenizer = Tokenizer.from_file(str(config.tokenizer.path))
    eos_id = tokenizer.token_to_id("<eos>")

    assert manifest == loaded_manifest
    assert manifest.dataset_requested_revision is None
    assert manifest.dataset_revision == f"injected-sha256:{manifest.dataset_fingerprint}"
    assert len(manifest.dataset_fingerprint) == 64
    assert sum(manifest.document_counts.values()) == 24
    assert set(manifest.artifacts) == {"train", "validation"}
    for split, artifact in manifest.artifacts.items():
        tokens = open_token_artifact(config.data.output_dir, artifact)
        assert tokens.tolist().count(eos_id) == manifest.document_counts[split]


def test_prepare_corpus_reuses_compatible_artifacts_before_consuming_source(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    documents = [
        {"id": f"doc-{index}", "text": f"sample document number {index}"}
        for index in range(24)
    ]
    first = prepare_corpus(config, documents=documents)

    def must_not_be_consumed():
        raise AssertionError("source was consumed before overwrite protection")
        yield

    reused = prepare_corpus(config, documents=must_not_be_consumed())

    assert reused == first
    assert load_manifest(config.data.output_dir / "manifest.json") == first


def test_prepare_corpus_refuses_incomplete_artifacts_before_consuming_source(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    documents = [
        {"id": f"doc-{index}", "text": f"sample document number {index}"}
        for index in range(24)
    ]
    manifest = prepare_corpus(config, documents=documents)
    (config.data.output_dir / manifest.artifacts["validation"].filename).unlink()

    def must_not_be_consumed():
        raise AssertionError("source was consumed before incomplete artifact validation")
        yield

    with pytest.raises(FileExistsError, match="incomplete"):
        prepare_corpus(config, documents=must_not_be_consumed())


def test_prepare_corpus_refuses_corrupt_artifacts_before_consuming_source(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    documents = [
        {"id": f"doc-{index}", "text": f"sample document number {index}"}
        for index in range(24)
    ]
    manifest = prepare_corpus(config, documents=documents)
    train_path = config.data.output_dir / manifest.artifacts["train"].filename
    data = train_path.read_bytes()
    train_path.write_bytes(bytes([data[0] ^ 0xFF]) + data[1:])

    def must_not_be_consumed():
        raise AssertionError("source was consumed before checksum validation")
        yield

    with pytest.raises(FileExistsError, match="invalid") as error:
        prepare_corpus(config, documents=must_not_be_consumed())

    assert isinstance(error.value.__cause__, ValueError)
    assert "checksum" in str(error.value.__cause__)


def test_prepare_corpus_refuses_incompatible_config_before_consuming_source(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path)
    documents = [
        {"id": f"doc-{index}", "text": f"sample document number {index}"}
        for index in range(24)
    ]
    prepare_corpus(config, documents=documents)
    incompatible_payload = config.model_dump(mode="python")
    incompatible_payload["data"]["validation_fraction"] = 0.25
    incompatible = TinyLLMConfig.model_validate(incompatible_payload)

    def must_not_be_consumed():
        raise AssertionError("source was consumed before config validation")
        yield

    with pytest.raises(FileExistsError, match="incompatible"):
        prepare_corpus(incompatible, documents=must_not_be_consumed())


def test_load_manifest_rejects_corrupt_artifact(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path)
    documents = [
        {"id": f"doc-{index}", "text": f"sample document number {index}"}
        for index in range(24)
    ]
    manifest = prepare_corpus(config, documents=documents)
    train_path = config.data.output_dir / manifest.artifacts["train"].filename
    data = train_path.read_bytes()
    train_path.write_bytes(bytes([data[0] ^ 0xFF]) + data[1:])

    with pytest.raises(ValueError, match="checksum"):
        load_manifest(config.data.output_dir / "manifest.json")


def test_injected_source_identity_is_deterministic_without_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_hub_call(*args, **kwargs):
        raise AssertionError("Hub must not be used for injected documents")

    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", unexpected_hub_call)
    documents = [
        {"id": f"doc-{index}", "text": f"stable sample {index}"}
        for index in range(24)
    ]
    first = prepare_corpus(_tiny_config(tmp_path / "first"), documents=documents)
    second = prepare_corpus(_tiny_config(tmp_path / "second"), documents=documents)

    assert first.dataset_revision == second.dataset_revision
    assert first.dataset_fingerprint == second.dataset_fingerprint


def test_hugging_face_source_loads_resolved_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_sha = "d" * 40
    calls: dict[str, object] = {}

    class FakeStreamingDataset(list):
        _fingerprint = "hub-fingerprint-123"

    def fake_dataset_info(self, repo_id, *, revision=None, **kwargs):
        calls["repo_id"] = repo_id
        calls["requested_revision"] = revision
        return SimpleNamespace(sha=resolved_sha)

    def fake_load_dataset(path, **kwargs):
        calls["loaded_revision"] = kwargs["revision"]
        return FakeStreamingDataset(
            {"id": f"doc-{index}", "text": f"hub sample {index}"}
            for index in range(24)
        )

    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", fake_dataset_info)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    manifest = prepare_corpus(_tiny_config(tmp_path))

    assert calls == {
        "repo_id": "codelion/fineweb-edu-100M",
        "requested_revision": None,
        "loaded_revision": resolved_sha,
    }
    assert manifest.dataset_requested_revision is None
    assert manifest.dataset_revision == resolved_sha
    assert manifest.dataset_fingerprint == "hub-fingerprint-123"


def _tiny_config(tmp_path: Path) -> TinyLLMConfig:
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
        {
            "vocab_size": 128,
            "min_frequency": 1,
            "path": tmp_path / "tokenizer.json",
        }
    )
    raw["model"]["vocab_size"] = 128
    return TinyLLMConfig.model_validate(raw)
