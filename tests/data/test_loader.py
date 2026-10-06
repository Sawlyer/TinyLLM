import shutil
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models

from tinyllm.data.artifacts import (
    MANIFEST_VERSION,
    CorpusManifest,
    load_manifest,
    sha256_file,
    write_manifest,
    write_token_artifact,
)
from tinyllm.data.loader import TokenDataset, create_dataloader

TOKENIZER_SHA256 = "a" * 64


@pytest.fixture()
def dataset_factory(tmp_path: Path):
    def factory(*, seed: int = 7, sequence_length: int = 4) -> TokenDataset:
        output_dir = tmp_path / "processed"
        output_dir.mkdir(exist_ok=True)
        train = np.arange(16, dtype=np.uint16)
        validation = np.arange(8, dtype=np.uint16)
        artifacts = {
            "train": write_token_artifact(
                output_dir,
                "train",
                train,
                tokenizer_sha256=TOKENIZER_SHA256,
                overwrite=True,
            ),
            "validation": write_token_artifact(
                output_dir,
                "validation",
                validation,
                tokenizer_sha256=TOKENIZER_SHA256,
                overwrite=True,
            ),
        }
        manifest = CorpusManifest(
            version=MANIFEST_VERSION,
            dataset_name="fixture",
            dataset_requested_revision=None,
            dataset_revision="injected-sha256:" + "b" * 64,
            dataset_fingerprint="fixture-fingerprint",
            text_field="text",
            seed=seed,
            validation_fraction=0.5,
            tokenizer_path=str(tmp_path / "tokenizer.json"),
            tokenizer_sha256=TOKENIZER_SHA256,
            tokenizer_config={"vocab_size": 32},
            data_config={
                "dataset_name": "fixture",
                "dataset_revision": None,
                "text_field": "text",
                "seed": seed,
                "validation_fraction": 0.5,
                "output_dir": str(output_dir),
            },
            document_counts={"train": 1, "validation": 1},
            artifacts=artifacts,
            tokenizer_vocab_size=16,
            special_token_ids={
                "<unk>": 0,
                "<bos>": 1,
                "<eos>": 2,
                "<pad>": 3,
            },
        )
        return TokenDataset(manifest, "train", sequence_length, seed, artifact_dir=output_dir)

    return factory


@pytest.fixture()
def dataset(dataset_factory):
    return dataset_factory()


def test_batch_targets_are_inputs_shifted_one_token(dataset: TokenDataset) -> None:
    x, y = dataset.batch(torch.tensor([0, 3]))

    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.tolist() == [[0, 1, 2, 3], [3, 4, 5, 6]]
    assert y.tolist() == [[1, 2, 3, 4], [4, 5, 6, 7]]


def test_seed_reproduces_sample_positions(dataset_factory) -> None:
    a = dataset_factory(seed=7).sample_positions(8)
    b = dataset_factory(seed=7).sample_positions(8)

    assert torch.equal(a, b)


def test_sample_positions_never_cross_final_target(dataset: TokenDataset) -> None:
    positions = dataset.sample_positions(256)

    assert int(positions.min()) >= 0
    assert int(positions.max()) <= len(dataset) - 1


def test_dataset_rejects_corpus_shorter_than_sequence_plus_target(
    dataset_factory,
) -> None:
    with pytest.raises(ValueError, match=r"sequence_length \+ 1"):
        dataset_factory(sequence_length=16)


def test_rng_state_can_resume_sampling(dataset_factory) -> None:
    original = dataset_factory(seed=9)
    original.sample_positions(3)
    state = original.get_rng_state()
    expected = original.sample_positions(5)

    resumed = TokenDataset(
        original.manifest,
        original.split,
        original.sequence_length,
        seed=9,
        artifact_dir=Path(original.manifest.data_config["output_dir"]),
    )
    resumed.set_rng_state(state)

    assert torch.equal(resumed.sample_positions(5), expected)


def test_dataloader_factory_uses_pinned_memory_only_with_accelerator(
    dataset: TokenDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader = create_dataloader(dataset, batch_size=2)
        x, y = next(iter(loader))

    assert loader.pin_memory is False
    assert x.shape == y.shape == (2, dataset.sequence_length)
    assert not [warning for warning in caught if "pin_memory" in str(warning.message)]


def test_relative_artifact_path_uses_manifest_location(
    dataset: TokenDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded, manifest_path = _write_nested_manifest(dataset, tmp_path)

    monkeypatch.chdir(tmp_path)
    loaded_dataset = TokenDataset(loaded, "train", sequence_length=4, seed=7)

    assert loaded_dataset.batch(torch.tensor([0]))[0].tolist() == [[0, 1, 2, 3]]


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_manifest_root_failure_is_not_masked_by_cwd_homonym(
    dataset: TokenDataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    loaded, manifest_path = _write_nested_manifest(dataset, tmp_path)
    canonical = manifest_path.parent / "train.bin"
    if failure == "missing":
        canonical.unlink()
    else:
        canonical.write_bytes(canonical.read_bytes()[:-1])
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    shutil.copy2(dataset.manifest.data_config["output_dir"] + "/train.bin", cwd / "train.bin")
    monkeypatch.chdir(cwd)

    with pytest.raises(ValueError, match="(missing|size mismatch|checksum)"):
        TokenDataset(loaded, "train", sequence_length=4, seed=7)


def test_unbound_manifest_requires_explicit_artifact_dir(
    dataset: TokenDataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = replace(
        dataset.manifest,
        data_config={**dataset.manifest.data_config, "output_dir": "."},
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    with pytest.raises(ValueError, match="artifact_dir"):
        TokenDataset(manifest, "train", sequence_length=4, seed=7)

    loaded = TokenDataset(
        manifest,
        "train",
        sequence_length=4,
        seed=7,
        artifact_dir=Path(dataset.manifest.data_config["output_dir"]),
    )
    assert loaded.batch(torch.tensor([0]))[0].tolist() == [[0, 1, 2, 3]]


def test_explicit_artifact_dir_allows_manifest_without_output_dir(
    dataset: TokenDataset,
) -> None:
    manifest = replace(
        dataset.manifest,
        data_config={
            key: value
            for key, value in dataset.manifest.data_config.items()
            if key != "output_dir"
        },
    )

    loaded = TokenDataset(
        manifest,
        "train",
        sequence_length=4,
        seed=7,
        artifact_dir=Path(dataset.manifest.data_config["output_dir"]),
    )

    assert loaded.batch(torch.tensor([0]))[0].tolist() == [[0, 1, 2, 3]]


def _write_nested_manifest(dataset: TokenDataset, tmp_path: Path):
    tokenizer_path = tmp_path / "tokenizer.json"
    vocabulary = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        **{f"token-{index}": index for index in range(4, 16)},
    }
    Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>")).save(str(tokenizer_path))
    tokenizer_sha256 = sha256_file(tokenizer_path)
    artifacts = {
        split: replace(artifact, tokenizer_sha256=tokenizer_sha256)
        for split, artifact in dataset.manifest.artifacts.items()
    }
    manifest = replace(
        dataset.manifest,
        tokenizer_path=str(tokenizer_path),
        tokenizer_sha256=tokenizer_sha256,
        data_config={**dataset.manifest.data_config, "output_dir": "."},
        artifacts=artifacts,
    )
    manifest_dir = tmp_path / "nested" / "processed"
    manifest_dir.mkdir(parents=True)
    source_dir = Path(dataset.manifest.data_config["output_dir"])
    for artifact in artifacts.values():
        shutil.copy2(source_dir / artifact.filename, manifest_dir / artifact.filename)
    manifest_path = manifest_dir / "manifest.json"
    write_manifest(manifest_path, manifest)
    return load_manifest(manifest_path), manifest_path
