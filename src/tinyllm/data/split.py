"""Stable document-level corpus splitting."""

import hashlib


def split_for_id(document_id: str, seed: int, validation_fraction: float) -> str:
    """Assign a document to a stable split using a seeded BLAKE2b digest."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    payload = f"{seed}\0{document_id}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"tinyllm-split").digest()
    sample = int.from_bytes(digest, "big") / 2**64
    return "validation" if sample < validation_fraction else "train"
