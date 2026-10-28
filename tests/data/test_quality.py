from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinyllm.cli import _build_parser
from tinyllm.config.load import load_config
from tinyllm.config.schema import TinyLLMConfig
from tinyllm.data.prepare import _spool_documents
from tinyllm.data.prepare import prepare_corpus
from tinyllm.data.quality import CorpusQualityOptions, StreamingCorpusQuality


def test_quality_report_counts_streamed_documents_and_filters_short_text(tmp_path):
    quality = StreamingCorpusQuality(
        tmp_path / "quality.sqlite",
        CorpusQualityOptions(min_characters=8),
    )
    try:
        assert quality.record("TinyLLM is useful.", "train")
        assert not quality.record("tiny", "validation")
        quality.record_invalid(blank=True)
        report = quality.report()
    finally:
        quality.close()

    assert report["source_records"] == 3
    assert report["blank_documents"] == 1
    assert report["splits"]["train"]["documents_after"] == 1
    assert report["splits"]["validation"]["filtered_reasons"] == {"too_short": 1}


def test_quality_filter_reports_exact_duplicates_across_splits(tmp_path):
    quality = StreamingCorpusQuality(
        tmp_path / "quality.sqlite",
        CorpusQualityOptions(deduplicate_exact=True),
    )
    try:
        assert quality.record("The same document appears here.", "train")
        assert not quality.record("The same document appears here.", "validation")
        report = quality.report()
    finally:
        quality.close()

    assert report["splits"]["validation"]["cross_split_exact_duplicates"] == 1
    assert report["splits"]["validation"]["filtered_reasons"] == {"exact_duplicate": 1}


def test_quality_near_deduplication_is_optional_and_hash_only(tmp_path):
    options = CorpusQualityOptions(deduplicate_near=True, near_duplicate_hamming_distance=7)
    quality = StreamingCorpusQuality(tmp_path / "quality.sqlite", options)
    try:
        first = "A helpful language model learns from carefully prepared text."
        second = "A helpful language model learns from carefully prepared texts."
        assert quality.record(first, "train")
        retained = quality.record(second, "train")
        report = quality.report()
    finally:
        quality.close()

    if not retained:
        assert report["near_duplicate_examples"]
        assert "document" not in report["near_duplicate_examples"][0]


def test_quality_options_reject_invalid_bounds():
    with pytest.raises(ValueError, match="cannot exceed"):
        CorpusQualityOptions(min_characters=20, max_characters=10)


def test_quality_detects_expected_language_and_repeated_runs(tmp_path):
    quality = StreamingCorpusQuality(
        tmp_path / "quality.sqlite",
        CorpusQualityOptions(
            expected_language="en",
            minimum_language_confidence=0.1,
            max_repeated_character_ratio=0.2,
        ),
    )
    try:
        assert not quality.record("bonjour le monde avec les amis", "train")
        assert not quality.record("the the the aaaaaaaaaa", "train")
        report = quality.report()
    finally:
        quality.close()

    reasons = report["splits"]["train"]["filtered_reasons"]
    assert reasons["unexpected_language"] == 1
    assert reasons["repeated_character_run"] == 1


def test_quality_can_filter_repeated_word_sequences(tmp_path):
    quality = StreamingCorpusQuality(
        tmp_path / "quality.sqlite",
        CorpusQualityOptions(max_repeated_ngram_ratio=0.25),
    )
    try:
        retained = quality.record(
            "the quiet fox runs across fields each morning " * 3,
            "train",
        )
        report = quality.report()
    finally:
        quality.close()

    assert not retained
    assert report["splits"]["train"]["filtered_reasons"] == {"repeated_word_sequences": 1}


def test_spooling_filters_after_deterministic_split_and_reports_counts(tmp_path):
    config = SimpleNamespace(
        data=SimpleNamespace(text_field="text", seed=42, validation_fraction=0.5)
    )
    documents = [
        {"id": "kept", "text": "This is a useful training example."},
        {"id": "short", "text": "tiny"},
        {"id": "blank", "text": "   "},
        None,
    ]

    counts, fingerprint, report = _spool_documents(
        documents,
        config,
        tmp_path,
        CorpusQualityOptions(min_characters=12),
    )

    assert len(fingerprint) == 64
    assert sum(counts.values()) == 1
    assert sum(
        len(path.read_text(encoding="utf-8").splitlines())
        for path in (tmp_path / "train.documents.jsonl", tmp_path / "validation.documents.jsonl")
    ) == 1
    assert report["source_records"] == 4
    assert report["blank_documents"] == 1
    assert report["invalid_records"] == 1
    assert sum(item["documents_filtered"] for item in report["splits"].values()) == 1


def test_prepare_publishes_report_and_rejects_changed_cached_filters(tmp_path):
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
        {"vocab_size": 128, "min_frequency": 1, "path": tmp_path / "tokenizer.json"}
    )
    raw["model"]["vocab_size"] = 128
    config = TinyLLMConfig.model_validate(raw)
    documents = [
        {"id": f"doc-{index}", "text": f"sample document number {index}"}
        for index in range(24)
    ] + [{"id": "short", "text": "tiny"}]
    options = CorpusQualityOptions(min_characters=12)

    manifest = prepare_corpus(config, documents=documents, quality_options=options)
    report_path = config.data.output_dir / "quality-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert sum(manifest.document_counts.values()) == 24
    assert report["filters"] == options.to_dict()
    assert report["source_records"] == 25
    assert sum(split["documents_before"] for split in report["splits"].values()) == 25
    assert sum(split["documents_after"] for split in report["splits"].values()) == 24
    assert prepare_corpus(config, documents=iter(()), quality_options=options) == manifest
    with pytest.raises(FileExistsError, match="different quality filters"):
        prepare_corpus(config, documents=iter(()))


def test_prepare_cli_exposes_opt_in_quality_filters():
    args = _build_parser().parse_args(
        [
            "prepare",
            "--min-document-characters",
            "100",
            "--deduplicate-exact",
            "--expected-language",
            "en",
        ]
    )

    assert args.min_document_characters == 100
    assert args.deduplicate_exact
    assert args.expected_language == "en"

