"""Project-trained BPE tokenizer."""

import os
from collections.abc import Iterable
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tinyllm.config.schema import TokenizerConfig

REQUIRED_SPECIAL_TOKENS = ("<unk>", "<bos>", "<eos>", "<pad>")


def train_tokenizer(
    texts: Iterable[str],
    config: TokenizerConfig,
    output: Path,
    *,
    overwrite: bool = False,
) -> Tokenizer:
    """Train a BPE tokenizer and atomically save it to ``output``."""
    missing = set(REQUIRED_SPECIAL_TOKENS).difference(config.special_tokens)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"tokenizer special_tokens missing required values: {missing_list}")

    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"tokenizer output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
        show_progress=False,
    )

    text_count = 0

    def counted_texts() -> Iterable[str]:
        nonlocal text_count
        for text in texts:
            if not isinstance(text, str) or not text:
                continue
            text_count += 1
            yield text

    tokenizer.train_from_iterator(counted_texts(), trainer=trainer)
    if text_count == 0:
        raise ValueError("cannot train tokenizer without training documents")

    temporary = output.with_name(f"{output.name}.tmp")
    try:
        temporary.unlink(missing_ok=True)
        tokenizer.save(str(temporary))
        with temporary.open("r+b") as tokenizer_file:
            os.fsync(tokenizer_file.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return tokenizer
