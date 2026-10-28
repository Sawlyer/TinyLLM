import torch

from tinyllm.cli import _build_parser
from tinyllm.config.schema import GenerationConfig, ModelConfig
from tinyllm.inference.benchmark import (
    benchmark_generation_cache,
    write_benchmark_report,
)
from tinyllm.model.transformer import TinyLLM


def tiny_model() -> TinyLLM:
    return TinyLLM(
        ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
            rope_theta=10_000.0,
            rms_norm_eps=1e-5,
        )
    ).eval()


def test_cache_benchmark_compares_greedy_outputs_at_equal_token_count():
    torch.manual_seed(7)
    model = tiny_model()
    prompt = torch.tensor([[3, 4, 5]])
    config = GenerationConfig(
        max_new_tokens=3,
        temperature=0.0,
        top_p=1.0,
        seed=23,
        eos_token_id=None,
    )

    result = benchmark_generation_cache(
        model, prompt, config, warmup_runs=1, measured_runs=2
    )

    assert result["greedy_outputs_equal"] is True
    assert result["prompt_tokens"] == 3
    assert result["generated_tokens"] == 3
    assert set(result["modes"]) == {"cached", "uncached"}
    assert result["modes"]["cached"]["time_to_first_token_ms"] > 0
    assert result["modes"]["uncached"]["decode_tokens_per_second"] > 0
    assert result["modes"]["cached"]["kv_cache_capacity_bytes"] == (
        2 * 2 * 2 * 4 * 8 * 4
    )


def test_cache_benchmark_rejects_prompt_and_generation_beyond_context():
    model = tiny_model()
    prompt = torch.tensor([[3, 4, 5, 6, 7, 8]])
    config = GenerationConfig(
        max_new_tokens=3,
        temperature=0.0,
        top_p=1.0,
        seed=23,
        eos_token_id=None,
    )

    try:
        benchmark_generation_cache(model, prompt, config, warmup_runs=0, measured_runs=1)
    except ValueError as error:
        assert "fit within model context" in str(error)
    else:
        raise AssertionError("expected context overflow to be rejected")


def test_benchmark_report_is_json_and_refuses_overwrite(tmp_path):
    output_path = tmp_path / "benchmark.json"
    write_benchmark_report({"greedy_outputs_equal": True}, output_path)

    assert '"greedy_outputs_equal": true' in output_path.read_text(encoding="utf-8")
    try:
        write_benchmark_report({"greedy_outputs_equal": False}, output_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected existing report to be preserved")


def test_benchmark_inference_cli_accepts_comparable_run_controls():
    args = _build_parser().parse_args(
        [
            "benchmark-inference",
            "--checkpoint",
            "model.pt",
            "--prompt",
            "Hello",
            "--max-new-tokens",
            "12",
            "--warmup-runs",
            "3",
            "--measured-runs",
            "7",
        ]
    )

    assert args.handler.__name__ == "_benchmark_inference_command"
    assert args.max_new_tokens == 12
    assert args.warmup_runs == 3
    assert args.measured_runs == 7

