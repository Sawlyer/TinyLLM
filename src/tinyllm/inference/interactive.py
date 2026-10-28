"""Small, local, one-prompt-at-a-time terminal generation session."""

from __future__ import annotations

import json
from collections.abc import Callable
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from tokenizers import Tokenizer

from tinyllm.config.schema import GenerationConfig, ModelConfig
from tinyllm.inference.generate import generate


class InteractiveSession:
    """Run independent generation prompts without persisting content by default."""

    _EDITABLE_PARAMETERS = {
        "max_new_tokens": int,
        "temperature": float,
        "top_k": int,
        "top_p": float,
        "repetition_penalty": float,
        "seed": int,
    }

    def __init__(
        self,
        *,
        model: Any,
        model_config: ModelConfig,
        tokenizer: Tokenizer,
        generation: GenerationConfig,
        device: torch.device,
        checkpoint: Path,
        use_cache: bool = True,
        history_file: Path | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.generation = generation
        self.device = device
        self.checkpoint = checkpoint
        self.use_cache = use_cache
        self.history_file = history_file
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.history: list[dict[str, Any]] = []

    def run(self) -> None:
        self.output_fn(f"Checkpoint : {self.checkpoint}")
        self.output_fn(f"Appareil : {self.device}")
        self.output_fn(
            "Chaque prompt repart de zéro. Le format conversationnel dépend des données "
            "d'entraînement du modèle."
        )
        self.output_fn("Commandes : /help, /set <paramètre> <valeur>, /history, /reset, /quit")
        while True:
            try:
                prompt = self.input_fn("tinyllm> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.output_fn("Fin de session.")
                break
            if not prompt:
                continue
            if prompt in {"/quit", "/exit"}:
                self.output_fn("Fin de session.")
                break
            if prompt == "/help":
                self._help()
                continue
            if prompt == "/history":
                self._show_history()
                continue
            if prompt == "/reset":
                self._reset_history()
                continue
            if prompt.startswith("/set"):
                self._set_parameter(prompt)
                continue
            if prompt.startswith("/"):
                self.output_fn("Commande inconnue. Tapez /help pour la liste.")
                continue
            try:
                self._generate(prompt)
            except KeyboardInterrupt:
                self.output_fn("\nGénération interrompue. Session toujours active.")
            except ValueError as error:
                self.output_fn(f"Génération impossible : {error}")

    def _generate(self, prompt: str) -> None:
        prompt_ids = self.tokenizer.encode(prompt).ids
        if not prompt_ids:
            self.output_fn("Prompt vide après tokenisation; génération ignorée.")
            return
        if min(prompt_ids) < 0 or max(prompt_ids) >= self.model_config.vocab_size:
            raise ValueError("prompt contains token IDs outside model vocab_size")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        self._synchronize()
        started_at = perf_counter()
        generated = generate(
            self.model,
            input_ids,
            self.generation,
            use_cache=self.use_cache,
        )
        self._synchronize()
        elapsed = max(0.0, perf_counter() - started_at)
        continuation_ids = generated[0, len(prompt_ids) :].tolist()
        stop_length = max(
            (
                len(sequence)
                for sequence in self.generation.stop_token_sequences
                if len(continuation_ids) >= len(sequence)
                and continuation_ids[-len(sequence) :] == sequence
            ),
            default=0,
        )
        if stop_length:
            del continuation_ids[-stop_length:]
        response = self.tokenizer.decode(continuation_ids, skip_special_tokens=False)
        tokens_per_second = len(continuation_ids) / elapsed if elapsed > 0 else 0.0
        self.output_fn(response or "(aucun nouveau texte)")
        self.output_fn(
            f"{len(continuation_ids)} tokens | {elapsed:.3f} s | {tokens_per_second:.2f} tokens/s"
        )
        record = {
            "prompt": prompt,
            "response": response,
            "generated_tokens": len(continuation_ids),
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_second,
        }
        self.history.append(record)
        if self.history_file is not None:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a", encoding="utf-8") as history_file:
                history_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _set_parameter(self, command: str) -> None:
        parts = command.split(maxsplit=2)
        if len(parts) != 3 or parts[1] not in self._EDITABLE_PARAMETERS:
            self.output_fn(
                "Usage : /set max_new_tokens|temperature|top_k|top_p|"
                "repetition_penalty|seed valeur"
            )
            return
        name, raw_value = parts[1], parts[2]
        if name == "top_k" and raw_value.lower() in {"none", "off"}:
            value = None
        else:
            try:
                value = self._EDITABLE_PARAMETERS[name](raw_value)
            except ValueError:
                self.output_fn(f"Valeur invalide pour {name} : {raw_value}")
                return
        if isinstance(value, float) and not isfinite(value):
            self.output_fn(f"Valeur invalide pour {name} : une valeur finie est requise")
            return
        values = self.generation.model_dump()
        values[name] = value
        try:
            updated = GenerationConfig.model_validate(values)
        except ValueError as error:
            self.output_fn(f"Paramètre refusé : {error}")
            return
        self.generation = updated
        self.output_fn(f"{name} = {value}")

    def _show_history(self) -> None:
        if not self.history:
            self.output_fn("Historique vide.")
            return
        for index, record in enumerate(self.history, start=1):
            self.output_fn(f"{index}. Vous : {record['prompt']}")
            self.output_fn(f"   TinyLLM : {record['response']}")
        self.output_fn(f"{len(self.history)} échanges en mémoire.")

    def _reset_history(self) -> None:
        self.history.clear()
        if self.history_file is not None:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text("", encoding="utf-8")
        self.output_fn("Historique de session effacé.")

    def _help(self) -> None:
        self.output_fn(
            "Texte : générer une réponse indépendante | /set <paramètre> <valeur> : "
            "modifier génération | /history : afficher session | /reset : effacer session "
            "| /quit : quitter"
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

