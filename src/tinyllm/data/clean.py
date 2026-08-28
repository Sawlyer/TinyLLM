"""Minimal deterministic document cleaning."""

import unicodedata


def normalize_document(text: str) -> str | None:
    """Trim and NFC-normalize a document, dropping missing or blank values."""
    if not isinstance(text, str):
        return None
    normalized = unicodedata.normalize("NFC", text).strip()
    return normalized or None
