from __future__ import annotations

from types import SimpleNamespace

from tinyllm.data.prepare import _spool_documents
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
    import pytest

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

