import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tinyllm import cli
from tinyllm.config.schema import GenerationConfig


def test_generate_cli_reports_effective_settings_and_per_prompt_metrics(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(device="cpu"),
        tokenizer=SimpleNamespace(path=tmp_path / "tokenizer.json"),
        generation=GenerationConfig(
            max_new_tokens=3,
            temperature=1.0,
            top_k=None,
            top_p=1.0,
            seed=42,
            eos_token_id=0,
        ),
        model=object(),
    )

    class FakeModelConfig:
        @classmethod
        def model_validate(cls, _value):
            return SimpleNamespace(vocab_size=16, model_dump=lambda **_kwargs: {})

    class FakeTokenizer:
        def encode(self, value):
            return SimpleNamespace(ids=[9] if value == "!" else [1, 2])

        def decode(self, values, *, skip_special_tokens):
            assert skip_special_tokens is False
            return "".join(f"<{token}>" for token in values)

    monkeypatch.setattr(cli, "_config_for_checkpoint", lambda *_args: config)
    monkeypatch.setattr(cli, "_resolve_inference_device", lambda _device: torch.device("cpu"))
    monkeypatch.setattr(
        cli,
        "load_model_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(
            model=object(),
            identity=SimpleNamespace(model_config={}, tokenizer_sha256="tokenizer-sha256"),
        ),
    )
    monkeypatch.setattr(cli, "ModelConfig", FakeModelConfig)
    monkeypatch.setattr(cli, "_require_model_config_match", lambda *_args: None)
    monkeypatch.setattr(cli, "sha256_file", lambda _path: "tokenizer-sha256")
    monkeypatch.setattr(cli, "_load_tokenizer", lambda _path: FakeTokenizer())
    monkeypatch.setattr(cli, "validate_tokenizer_for_model", lambda *_args: None)

    def fake_generate(_model, prompt_ids, _generation, *, use_cache, on_first_token):
        assert use_cache is True
        on_first_token()
        return torch.cat((prompt_ids, torch.tensor([[7, 9]])), dim=1)

    monkeypatch.setattr(cli, "generate", fake_generate)
    args = cli._build_parser().parse_args(
        [
            "generate",
            "--checkpoint",
            "model.pt",
            "--prompt",
            "first prompt",
            "--prompt",
            "second prompt",
            "--stop",
            "!",
            "--repetition-penalty",
            "1.25",
            "--json",
        ]
    )

    cli._generate_command(args)

    report = json.loads(capsys.readouterr().out)
    assert report["parameters"]["repetition_penalty"] == 1.25
    assert report["parameters"]["stop_token_sequences"] == [[9]]
    assert len(report["results"]) == 2
    assert report["results"][0]["continuation"] == "<7>"
    assert report["results"][0]["generated_tokens"] == 1
    assert report["results"][0]["tokens_per_second"] > 0
    assert report["results"][0]["first_token_latency_ms"] is not None

