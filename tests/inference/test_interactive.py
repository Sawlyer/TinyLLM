import json
from pathlib import Path
from types import SimpleNamespace

import torch

from tinyllm.config.schema import GenerationConfig
from tinyllm.inference.interactive import InteractiveSession


class FakeTokenizer:
    def encode(self, text: str) -> SimpleNamespace:
        return SimpleNamespace(ids=[9] if text == "invalid" else [1] if text else [])

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return " ".join(str(token_id) for token_id in token_ids)


def make_session(inputs, output, **kwargs) -> InteractiveSession:
    model = SimpleNamespace(config=SimpleNamespace(max_seq_len=8, vocab_size=8))
    config = GenerationConfig(
        max_new_tokens=2,
        temperature=1.0,
        top_k=None,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=7,
        eos_token_id=0,
    )
    return InteractiveSession(
        model=model,
        model_config=model.config,
        tokenizer=FakeTokenizer(),
        generation=config,
        device=torch.device("cpu"),
        checkpoint=Path("model.pt"),
        input_fn=lambda _prompt: next(inputs),
        output_fn=output.append,
        **kwargs,
    )


def test_interactive_session_updates_settings_and_keeps_history_in_memory(monkeypatch) -> None:
    inputs = iter(["first", "/set temperature 0.25", "second", "/history", "/quit"])
    output: list[str] = []
    observed_temperatures: list[float] = []

    def fake_generate(_model, input_ids, generation, *, use_cache):
        observed_temperatures.append(generation.temperature)
        assert use_cache is True
        return torch.cat((input_ids, torch.tensor([[2]])), dim=1)

    monkeypatch.setattr("tinyllm.inference.interactive.generate", fake_generate)
    session = make_session(inputs, output)

    session.run()

    assert observed_temperatures == [1.0, 0.25]
    assert len(session.history) == 2
    assert any("Vous : first" in line for line in output)
    assert any("TinyLLM : 2" in line for line in output)
    assert any("2 échanges" in line for line in output)


def test_interactive_history_is_written_only_when_file_is_explicit(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_generate(_model, input_ids, _generation, *, use_cache):
        return torch.cat((input_ids, torch.tensor([[3]])), dim=1)

    monkeypatch.setattr("tinyllm.inference.interactive.generate", fake_generate)
    output: list[str] = []
    no_file_session = make_session(iter(["private prompt", "/quit"]), output)

    no_file_session.run()

    assert no_file_session.history
    assert not (tmp_path / "history.jsonl").exists()

    history_path = tmp_path / "history.jsonl"
    saved_session = make_session(
        iter(["saved prompt", "/quit"]),
        output,
        history_file=history_path,
    )

    saved_session.run()

    rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["prompt"] == "saved prompt"
    assert rows[0]["response"] == "3"


def test_interactive_reset_clears_session_history(monkeypatch) -> None:
    def fake_generate(_model, input_ids, _generation, *, use_cache):
        return torch.cat((input_ids, torch.tensor([[4]])), dim=1)

    monkeypatch.setattr("tinyllm.inference.interactive.generate", fake_generate)
    output: list[str] = []
    session = make_session(iter(["prompt", "/reset", "/history", "/quit"]), output)

    session.run()

    assert session.history == []
    assert any("Historique de session effacé" in line for line in output)


def test_interactive_prompt_loop_handles_eof_and_keyboard_interrupt(monkeypatch) -> None:
    def interrupt_generation(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("tinyllm.inference.interactive.generate", interrupt_generation)
    output: list[str] = []
    session = make_session(iter(["prompt", "/quit"]), output)

    session.run()

    assert session.history == []
    assert any("Génération interrompue" in line for line in output)


def test_interactive_session_omits_configured_stop_sequence(monkeypatch) -> None:
    def fake_generate(_model, input_ids, _generation, *, use_cache):
        return torch.cat((input_ids, torch.tensor([[2, 0, 0]])), dim=1)

    monkeypatch.setattr("tinyllm.inference.interactive.generate", fake_generate)
    output: list[str] = []
    session = make_session(iter(["prompt", "/quit"]), output)
    session.generation = GenerationConfig.model_validate(
        {**session.generation.model_dump(), "stop_token_sequences": [[0, 0]]}
    )

    session.run()

    assert session.history[0]["response"] == "2"
    assert session.history[0]["generated_tokens"] == 1


def test_interactive_session_keeps_running_after_invalid_prompt(monkeypatch) -> None:
    def fake_generate(_model, input_ids, _generation, *, use_cache):
        return torch.cat((input_ids, torch.tensor([[5]])), dim=1)

    monkeypatch.setattr("tinyllm.inference.interactive.generate", fake_generate)
    output: list[str] = []
    session = make_session(iter(["invalid", "valid", "/quit"]), output)

    session.run()

    assert [record["prompt"] for record in session.history] == ["valid"]
    assert any("Génération impossible" in line for line in output)


def test_interactive_parser_accepts_checkpoint_and_optional_history() -> None:
    from tinyllm.cli import _build_parser

    args = _build_parser().parse_args(
        ["interactive", "--checkpoint", "model.pt", "--history-file", "history.jsonl"]
    )

    assert args.command == "interactive"
    assert args.checkpoint == Path("model.pt")
    assert args.history_file == Path("history.jsonl")

