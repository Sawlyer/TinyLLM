"""Autoregressive text-token generation."""

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
) -> Tensor:
    """Generate tokens, rolling the active context window when it fills."""
    if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("prompt must have shape [non-empty batch, non-empty sequence]")

    context_length = model.config.max_seq_len
    if input_ids.shape[1] > context_length:
        raise ValueError(f"prompt length exceeds model context of {context_length} tokens")

    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(config.seed)
    generated = input_ids.clone()
    caches: list[KVCache] | None = None
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    eos_token_id = config.eos_token_id
    if eos_token_id is not None:
        if not isinstance(eos_token_id, int) or not 0 <= eos_token_id < model.config.vocab_size:
            raise ValueError("eos_token_id must identify a token in the model vocabulary")

    training_states = [(module, module.training) for module in model.modules()]
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
            )
            if eos_token_id is not None:
                eos_fill = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished[:, None], eos_fill, next_token)

            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(eos_token_id)
                if bool(finished.all()):
                    break
    finally:
        for module, training in training_states:
            module.training = training

    return generated
