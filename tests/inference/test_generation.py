from types import SimpleNamespace

import pytest
import torch

from tinyllm.config.schema import GenerationConfig, ModelConfig
from tinyllm.inference.generate import generate
from tinyllm.model.cache import KVCache
from tinyllm.model.transformer import TinyLLM


@pytest.fixture
def tiny_model() -> TinyLLM:
    torch.manual_seed(7)
    model = TinyLLM(
        ModelConfig(
            vocab_size=64,
            max_seq_len=6,
            d_model=16,
            n_layers=2,
            n_heads=8,
            n_kv_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            rope_theta=10_000.0,
            rms_norm_eps=1e-5,
        )
    )
    model.eval()
    return model


def greedy_config(max_new_tokens: int) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        seed=11,
    )


def force_greedy_token(model: TinyLLM, token_id: int) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.final_norm.weight.fill_(1)
        model.token_embedding.weight[1, 0] = 1
        model.token_embedding.weight[token_id, 0] = 2


def mixed_training_flags(model: TinyLLM) -> list[bool]:
    model.train()
    model.blocks[0].attention.eval()
    return [module.training for module in model.modules()]


def test_kv_cache_appends_key_and_value_sequences() -> None:
    cache = KVCache(capacity=6)
    first_key = torch.randn(2, 4, 3, 2)
    first_value = torch.randn(2, 4, 3, 2)
    next_key = torch.randn(2, 4, 1, 2)
    next_value = torch.randn(2, 4, 1, 2)

    cache.append(first_key, first_value)
    assert cache.key is not None
    assert cache.value is not None
    key_data_ptr = cache.key.data_ptr()
    value_data_ptr = cache.value.data_ptr()
    cache.append(next_key, next_value)

    assert cache.capacity == 6
    assert cache.length == 4
    assert cache.key is not None
    assert cache.value is not None
    assert cache.key.data_ptr() == key_data_ptr
    assert cache.value.data_ptr() == value_data_ptr
    assert torch.equal(cache.key, torch.cat((first_key, next_key), dim=2))
    assert torch.equal(cache.value, torch.cat((first_value, next_value), dim=2))


@pytest.mark.parametrize("via_append", [False, True])
def test_kv_cache_rejects_key_value_dtype_mismatch(via_append: bool) -> None:
    key = torch.zeros(1, 2, 1, 4, dtype=torch.float32)
    value = torch.zeros(1, 2, 1, 4, dtype=torch.float64)

    with pytest.raises(ValueError, match="same dtype"):
        if via_append:
            KVCache().append(key, value)
        else:
            KVCache(key=key, value=value)


@pytest.mark.parametrize("via_append", [False, True])
def test_kv_cache_rejects_key_value_device_mismatch(via_append: bool) -> None:
    key = torch.zeros(1, 2, 1, 4)
    value = torch.empty(1, 2, 1, 4, device="meta")

    with pytest.raises(ValueError, match="same device"):
        if via_append:
            KVCache().append(key, value)
        else:
            KVCache(key=key, value=value)


def test_cached_model_uses_offset_positions_and_mask(tiny_model: TinyLLM) -> None:
    ids = torch.tensor([[1, 4, 7, 3, 9]])
    full = tiny_model(ids).logits
    caches = [KVCache() for _ in tiny_model.blocks]

    primed = tiny_model(ids[:, :3], caches=caches)
    assert primed.caches is not None
    first_decoded = tiny_model(ids[:, 3:4], caches=primed.caches)
    assert first_decoded.caches is not None
    decoded = tiny_model(ids[:, 4:], caches=first_decoded.caches)

    assert decoded.caches is not None
    assert [cache.length for cache in decoded.caches] == [5, 5]
    assert torch.allclose(first_decoded.logits, full[:, 3:4], atol=1e-5, rtol=1e-5)
    assert torch.allclose(decoded.logits, full[:, 4:], atol=1e-5, rtol=1e-5)


def test_greedy_cached_generation_matches_uncached_across_context_rollover(
    tiny_model: TinyLLM,
) -> None:
    prompt = torch.tensor([[1, 4, 7, 3, 9]])

    cached = generate(tiny_model, prompt, greedy_config(5), use_cache=True)
    plain = generate(tiny_model, prompt, greedy_config(5), use_cache=False)

    assert cached.shape == (1, 10)
    assert torch.equal(cached, plain)


def test_cached_generation_primes_then_passes_only_newest_token(tiny_model: TinyLLM) -> None:
    seen_lengths: list[int] = []

    def record_input_length(_module: TinyLLM, args: tuple[torch.Tensor, ...]) -> None:
        seen_lengths.append(args[0].shape[1])

    handle = tiny_model.register_forward_pre_hook(record_input_length)
    try:
        generate(tiny_model, torch.tensor([[1, 4, 7]]), greedy_config(3), use_cache=True)
    finally:
        handle.remove()

    assert seen_lengths == [3, 1, 1]


def test_generation_stops_after_eos(tiny_model: TinyLLM) -> None:
    config = GenerationConfig(
        max_new_tokens=5,
        temperature=0.0,
        top_k=None,
        top_p=1.0,
        seed=2,
        eos_token_id=0,
    )
    with torch.no_grad():
        for parameter in tiny_model.parameters():
            parameter.zero_()

    generated = generate(tiny_model, torch.tensor([[1, 4]]), config)

    assert torch.equal(generated, torch.tensor([[1, 4, 0]]))


def test_generation_config_exposes_optional_eos_without_assuming_token_two(
    tiny_model: TinyLLM,
) -> None:
    config = greedy_config(3)
    force_greedy_token(tiny_model, 2)

    generated = generate(tiny_model, torch.tensor([[1]]), config)

    assert config.eos_token_id is None
    assert torch.equal(generated, torch.tensor([[1, 2, 2, 2]]))


def test_generation_restores_each_module_training_flag(tiny_model: TinyLLM) -> None:
    expected_flags = mixed_training_flags(tiny_model)

    generate(tiny_model, torch.tensor([[1, 4]]), greedy_config(2))

    assert [module.training for module in tiny_model.modules()] == expected_flags


def test_generation_restores_each_module_training_flag_after_error(tiny_model: TinyLLM) -> None:
    expected_flags = mixed_training_flags(tiny_model)
    invalid_config = SimpleNamespace(
        max_new_tokens=2,
        temperature=-1.0,
        top_k=None,
        top_p=1.0,
        seed=2,
        eos_token_id=None,
    )

    with pytest.raises(ValueError, match="temperature"):
        generate(tiny_model, torch.tensor([[1, 4]]), invalid_config)

    assert [module.training for module in tiny_model.modules()] == expected_flags


def test_generation_rejects_prompt_longer_than_context(tiny_model: TinyLLM) -> None:
    prompt = torch.ones((1, tiny_model.config.max_seq_len + 1), dtype=torch.long)

    with pytest.raises(ValueError, match="prompt.*context"):
        generate(tiny_model, prompt, greedy_config(1))
