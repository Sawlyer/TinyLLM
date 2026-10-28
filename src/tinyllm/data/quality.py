"""Bounded-memory corpus quality measurement and optional filtering."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_LANGUAGE_MARKERS = {
    "de": frozenset("der die das und ist zu von mit nicht ein eine auf für ich du wir".split()),
    "en": frozenset("the and is to of with not a an on for i you we this that".split()),
    "es": frozenset("el la los las y es de con no un una en para yo tú que".split()),
    "fr": frozenset("le la les et est de avec pas un une en pour je tu nous que".split()),
}
_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_REPEAT_PERSON = b"TinyLLMQ"


@dataclass(frozen=True)
class CorpusQualityOptions:
    min_characters: int | None = None
    max_characters: int | None = None
    max_repeated_character_ratio: float | None = None
    max_repeated_ngram_ratio: float | None = None
    expected_language: str | None = None
    minimum_language_confidence: float = 0.65
    deduplicate_exact: bool = False
    deduplicate_near: bool = False
    near_duplicate_hamming_distance: int = 3

    def __post_init__(self) -> None:
        if self.min_characters is not None and self.min_characters < 1:
            raise ValueError("min_characters must be positive")
        if self.max_characters is not None and self.max_characters < 1:
            raise ValueError("max_characters must be positive")
        if (
            self.min_characters is not None
            and self.max_characters is not None
            and self.min_characters > self.max_characters
        ):
            raise ValueError("min_characters cannot exceed max_characters")
        if self.max_repeated_character_ratio is not None and not (
            0.0 <= self.max_repeated_character_ratio <= 1.0
        ):
            raise ValueError("max_repeated_character_ratio must be between 0 and 1")
        if self.max_repeated_ngram_ratio is not None and not (
            0.0 <= self.max_repeated_ngram_ratio <= 1.0
        ):
            raise ValueError("max_repeated_ngram_ratio must be between 0 and 1")
        if self.expected_language not in {None, *_LANGUAGE_MARKERS}:
            raise ValueError("expected_language must be one of de, en, es, fr")
        if not 0.0 <= self.minimum_language_confidence <= 1.0:
            raise ValueError("minimum_language_confidence must be between 0 and 1")
        if not 0 <= self.near_duplicate_hamming_distance <= 7:
            raise ValueError("near_duplicate_hamming_distance must be between 0 and 7")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_characters": self.min_characters,
            "max_characters": self.max_characters,
            "max_repeated_character_ratio": self.max_repeated_character_ratio,
            "max_repeated_ngram_ratio": self.max_repeated_ngram_ratio,
            "expected_language": self.expected_language,
            "minimum_language_confidence": self.minimum_language_confidence,
            "deduplicate_exact": self.deduplicate_exact,
            "deduplicate_near": self.deduplicate_near,
            "near_duplicate_hamming_distance": self.near_duplicate_hamming_distance,
        }


class _SplitStats:
    def __init__(self) -> None:
        self.before = 0
        self.after = 0
        self.filtered_reasons: Counter[str] = Counter()
        self.duplicate_exact = 0
        self.duplicate_near = 0
        self.cross_split_exact = 0
        self.cross_split_near = 0
        self.characters_before = 0
        self.characters_after = 0
        self.min_characters: int | None = None
        self.max_characters = 0
        self.languages_before: Counter[str] = Counter()
        self.languages_after: Counter[str] = Counter()

    def observe_before(self, characters: int, language: str) -> None:
        self.before += 1
        self.characters_before += characters
        self.min_characters = (
            characters if self.min_characters is None else min(self.min_characters, characters)
        )
        self.max_characters = max(self.max_characters, characters)
        self.languages_before[language] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_before": self.before,
            "documents_after": self.after,
            "documents_filtered": self.before - self.after,
            "characters_before": self.characters_before,
            "characters_after": self.characters_after,
            "mean_characters_before": self.characters_before / self.before if self.before else 0.0,
            "mean_characters_after": self.characters_after / self.after if self.after else 0.0,
            "min_characters": self.min_characters or 0,
            "max_characters": self.max_characters,
            "language_counts_before": dict(sorted(self.languages_before.items())),
            "language_counts_after": dict(sorted(self.languages_after.items())),
            "exact_duplicates": self.duplicate_exact,
            "near_duplicates": self.duplicate_near,
            "cross_split_exact_duplicates": self.cross_split_exact,
            "cross_split_near_duplicates": self.cross_split_near,
            "filtered_reasons": dict(sorted(self.filtered_reasons.items())),
        }


class StreamingCorpusQuality:
    """Measure normalized documents while keeping duplicate indexes on disk."""

    def __init__(self, database_path: Path, options: CorpusQualityOptions) -> None:
        self.options = options
        self.source_records = 0
        self.invalid_records = 0
        self.blank_documents = 0
        self.splits = {"train": _SplitStats(), "validation": _SplitStats()}
        self.near_duplicate_examples: list[dict[str, object]] = []
        self.database = sqlite3.connect(database_path)
        self.database.execute("PRAGMA journal_mode=OFF")
        self.database.execute("PRAGMA synchronous=OFF")
        self.database.execute("PRAGMA temp_store=FILE")
        self.database.execute("PRAGMA cache_size=-8192")
        self.database.executescript(
            """
            CREATE TABLE seen_text (
                digest TEXT PRIMARY KEY,
                split TEXT NOT NULL
            );
            CREATE TABLE signatures (
                digest TEXT PRIMARY KEY,
                signature BLOB NOT NULL
            );
            CREATE TABLE signature_bands (
                band INTEGER NOT NULL,
                value INTEGER NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY (band, value, digest)
            );
            CREATE INDEX signature_bands_lookup ON signature_bands (band, value);
            """
        )

    def close(self) -> None:
        self.database.commit()
        self.database.close()

    def record_invalid(self, *, blank: bool = False) -> None:
        self.source_records += 1
        if blank:
            self.blank_documents += 1
        else:
            self.invalid_records += 1

    def record(self, text: str, split: str) -> bool:
        """Count one normalized document and return whether it should be kept."""
        self.source_records += 1
        stats = self.splits[split]
        repeated_ratio = _max_repeated_character_ratio(text)
        repeated_ngram_ratio = _repeated_word_ngram_ratio(text)
        language, confidence = detect_language(text)
        stats.observe_before(len(text), language)

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seen = self.database.execute(
            "SELECT split FROM seen_text WHERE digest = ?", (digest,)
        ).fetchone()
        exact_duplicate = seen is not None
        cross_split_exact = exact_duplicate and seen[0] != split
        if not exact_duplicate:
            self.database.execute(
                "INSERT INTO seen_text (digest, split) VALUES (?, ?)", (digest, split)
            )

        signature = _simhash(text)
        near_match = None if exact_duplicate else self._find_near_match(signature)
        near_duplicate = near_match is not None
        cross_split_near = near_duplicate and near_match[1] != split
        if exact_duplicate:
            stats.duplicate_exact += 1
            if cross_split_exact:
                stats.cross_split_exact += 1
        elif near_duplicate:
            stats.duplicate_near += 1
            if cross_split_near:
                stats.cross_split_near += 1
            if len(self.near_duplicate_examples) < 10:
                self.near_duplicate_examples.append(
                    {
                        "split": split,
                        "matched_split": near_match[1],
                        "hamming_distance": near_match[0],
                        "document_sha256_prefix": digest[:12],
                        "matched_sha256_prefix": near_match[2][:12],
                    }
                )

        reasons = self._filter_reasons(
            text=text,
            repeated_ratio=repeated_ratio,
            repeated_ngram_ratio=repeated_ngram_ratio,
            language=language,
            language_confidence=confidence,
            exact_duplicate=exact_duplicate,
            near_duplicate=near_duplicate,
        )
        if reasons:
            stats.filtered_reasons.update(reasons)
            return False

        stats.after += 1
        stats.characters_after += len(text)
        stats.languages_after[language] += 1
        if not exact_duplicate:
            self._store_signature(digest, signature)
        return True

    def _filter_reasons(
        self,
        *,
        text: str,
        repeated_ratio: float,
        repeated_ngram_ratio: float,
        language: str,
        language_confidence: float,
        exact_duplicate: bool,
        near_duplicate: bool,
    ) -> list[str]:
        reasons = []
        if self.options.min_characters is not None and len(text) < self.options.min_characters:
            reasons.append("too_short")
        if self.options.max_characters is not None and len(text) > self.options.max_characters:
            reasons.append("too_long")
        if (
            self.options.max_repeated_character_ratio is not None
            and repeated_ratio > self.options.max_repeated_character_ratio
        ):
            reasons.append("repeated_character_run")
        if (
            self.options.max_repeated_ngram_ratio is not None
            and repeated_ngram_ratio > self.options.max_repeated_ngram_ratio
        ):
            reasons.append("repeated_word_sequences")
        if (
            self.options.expected_language is not None
            and language_confidence >= self.options.minimum_language_confidence
            and language != self.options.expected_language
        ):
            reasons.append("unexpected_language")
        if self.options.deduplicate_exact and exact_duplicate:
            reasons.append("exact_duplicate")
        if self.options.deduplicate_near and near_duplicate:
            reasons.append("near_duplicate")
        return reasons

    def _find_near_match(self, signature: int) -> tuple[int, str, str] | None:
        candidates: set[str] = set()
        for band, value in enumerate(_signature_bands(signature)):
            rows = self.database.execute(
                "SELECT digest FROM signature_bands WHERE band = ? AND value = ? LIMIT 1001",
                (band, value),
            )
            candidates.update(row[0] for row in rows)
            if len(candidates) > 1000:
                break
        best: tuple[int, str, str] | None = None
        for digest in sorted(candidates):
            row = self.database.execute(
                "SELECT signatures.signature, seen_text.split "
                "FROM signatures JOIN seen_text USING (digest) WHERE digest = ?",
                (digest,),
            ).fetchone()
            if row is None:
                continue
            prior_signature = int.from_bytes(row[0], "big")
            distance = (signature ^ prior_signature).bit_count()
            if distance <= self.options.near_duplicate_hamming_distance and (
                best is None or (distance, digest) < (best[0], best[2])
            ):
                best = (distance, row[1], digest)
        return best

    def _store_signature(self, digest: str, signature: int) -> None:
        encoded = signature.to_bytes(8, "big")
        self.database.execute(
            "INSERT INTO signatures (digest, signature) VALUES (?, ?)", (digest, encoded)
        )
        self.database.executemany(
            "INSERT INTO signature_bands (band, value, digest) VALUES (?, ?, ?)",
            [
                (band, value, digest)
                for band, value in enumerate(_signature_bands(signature))
            ],
        )

    def report(self) -> dict[str, Any]:
        return {
            "algorithm": {
                "text_normalization": "NFC and trim before measurements",
                "exact_duplicates": "SHA-256 of normalized UTF-8 text",
                "near_duplicates": (
                    "64-bit SimHash of character 5-grams and word bigrams; "
                    "eight 8-bit lookup bands; Hamming threshold up to 7"
                ),
                "language_detection": "deterministic English/French/German/Spanish marker heuristic",
                "language_filter_behavior": (
                    "remove only when expected language differs and confidence meets threshold"
                ),
                "near_duplicate_candidate_cap": 1000,
                "near_duplicate_is_approximate": True,
                "memory": "SQLite disk index; bounded page cache; stream order defines retained duplicate",
            },
            "filters": self.options.to_dict(),
            "source_records": self.source_records,
            "blank_documents": self.blank_documents,
            "invalid_records": self.invalid_records,
            "splits": {split: stats.to_dict() for split, stats in self.splits.items()},
            "near_duplicate_examples": list(self.near_duplicate_examples),
        }


def detect_language(text: str) -> tuple[str, float]:
    words = _WORD_PATTERN.findall(text.casefold())
    if len(words) < 3:
        return "unknown", 0.0
    counts = {
        language: sum(word in markers for word in words)
        for language, markers in _LANGUAGE_MARKERS.items()
    }
    language, hits = max(counts.items(), key=lambda item: (item[1], item[0]))
    confidence = hits / len(words)
    if hits < 2 or confidence < 0.08:
        return "unknown", confidence
    return language, confidence


def _max_repeated_character_ratio(text: str) -> float:
    if not text:
        return 0.0
    longest = run = 1
    previous = text[0]
    for character in text[1:]:
        run = run + 1 if character == previous else 1
        longest = max(longest, run)
        previous = character
    return longest / len(text)


def _repeated_word_ngram_ratio(text: str) -> float:
    words = _WORD_PATTERN.findall(text.casefold())
    if len(words) < 4:
        return 0.0
    ngrams = Counter(tuple(words[index : index + 3]) for index in range(len(words) - 2))
    repeated_words = sum((frequency - 1) * 3 for frequency in ngrams.values() if frequency > 1)
    return min(1.0, repeated_words / len(words))


def _simhash(text: str) -> int:
    normalized = text.casefold()
    words = _WORD_PATTERN.findall(normalized)
    shingles = [f"c:{normalized[index:index + 5]}" for index in range(max(1, len(normalized) - 4))]
    shingles.extend(f"w:{left} {right}" for left, right in zip(words, words[1:]))
    weights = [0] * 64
    for shingle, frequency in Counter(shingles).items():
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8, person=_REPEAT_PERSON)
        value = int.from_bytes(digest.digest(), "big")
        for bit in range(64):
            weights[bit] += frequency if value & (1 << bit) else -frequency
    signature = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            signature |= 1 << bit
    return signature


def _signature_bands(signature: int) -> tuple[int, ...]:
    return tuple((signature >> (band * 8)) & 0xFF for band in range(8))

