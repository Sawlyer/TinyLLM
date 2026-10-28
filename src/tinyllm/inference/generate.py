"""Autoregressive text-token generation."""

from collections.abc import Callable

import torch
from torch import Tensor

from tinyllm.config.schema import GenerationConfig
from tinyllm.inference.sampling import sample_next
from tinyllm.model.cache import KVCache
from tinyllm.model.transformer import TinyLLM


@torch.inference_mode()
def generate(
    model: TinyLLM,
    input_ids: Tensor,
    config: GenerationConfig,
    use_cache: bool = True,
    on_first_token: Callable[[], None] | None = None,
) -> Tensor:
    """Generate tokens, rolling the active context window when it fills."""
    if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("prompt must have shape [non-empty batch, non-empty sequence]")

    context_length = model.config.max_seq_len
    if input_ids.shape[1] > context_length:
        raise ValueError(f"prompt length exceeds model context of {context_length} tokens")
    if input_ids.dtype != torch.long:
        raise ValueError("prompt token IDs must use torch.long")
    if bool((input_ids < 0).any()) or bool((input_ids >= model.config.vocab_size).any()):
        raise ValueError("prompt contains token IDs outside model vocab_size")

    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(config.seed)
    generated = input_ids.clone()
    caches: list[KVCache] | None = None
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    eos_token_id = config.eos_token_id
    if eos_token_id is not None:
        if not isinstance(eos_token_id, int) or not 0 <= eos_token_id < model.config.vocab_size:
            raise ValueError("eos_token_id must identify a token in the model vocabulary")
    stop_sequences: list[Tensor] = []
    for sequence in getattr(config, "stop_token_sequences", ()):
        if not sequence or any(type(token_id) is not int for token_id in sequence):
            raise ValueError("stop token sequences must contain at least one integer token ID")
        if any(token_id < 0 or token_id >= model.config.vocab_size for token_id in sequence):
            raise ValueError("stop token sequence contains a token outside model vocab_size")
        stop_sequences.append(torch.tensor(sequence, dtype=torch.long, device=input_ids.device))

    training_states = [(module, module.training) for module in model.modules()]
    generated_tokens = 0
    try:
        model.eval()
        for _ in range(config.max_new_tokens):
            context = generated[:, -context_length:]
            if use_cache:
                must_prime = caches is None or caches[0].length >= context_length
                if must_prime:
                    caches = [KVCache(capacity=context_length) for _ in model.blocks]
                    model_input = context
                else:
                    model_input = generated[:, -1:]
                output = model(model_input, caches=caches)
                if output.caches is None:
                    raise RuntimeError("model did not return caches during cached generation")
                caches = output.caches
            else:
                output = model(context)

            next_token = sample_next(
                output.logits[:, -1],
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                generator=generator,
                input_ids=generated,
                repetition_penalty=getattr(config, "repetition_penalty", 1.0),
            )
            if generated_tokens == 0 and on_first_token is not None:
                on_first_token()
            fill_token_id = eos_token_id if eos_token_id is not None else 0
            finished_fill = torch.full_like(next_token, fill_token_id)
            next_token = torch.where(finished[:, None], finished_fill, next_token)

            generated = torch.cat((generated, next_token), dim=1)
            generated_tokens += 1
            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(eos_token_id)
            for sequence in stop_sequences:
                sequence_length = sequence.numel()
                if generated_tokens >= sequence_length:
                    matches = generated[:, -sequence_length:].eq(sequence).all(dim=1)
                    finished |= matches
            if bool(finished.all()):
                break
    finally:
        for module, training in training_states:
            module.training = training

    return generated

