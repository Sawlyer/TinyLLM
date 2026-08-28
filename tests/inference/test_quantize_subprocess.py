from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _write_fake_torchao(root: Path) -> None:
    package = root / "torchao"
    package.mkdir()
    (package / "__init__.py").write_text(
        '__version__ = "test-1.0"\n',
        encoding="utf-8",
    )
    (package / "quantization.py").write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            import torch


            class ReloadTensor(torch.Tensor):
                @staticmethod
                def __new__(cls, value):
                    return torch.Tensor._make_subclass(cls, value.detach(), False)


            @dataclass
            class Int8WeightOnlyConfig:
                bits: int = 8


            def quantize_(model, config):
                del config
                for module in model.modules():
                    if isinstance(module, torch.nn.Linear):
                        module.weight = torch.nn.Parameter(
                            ReloadTensor(module.weight.detach()),
                            requires_grad=False,
                        )
            """
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("recipe", ["int8-weight-only"])
def test_torchao_export_reloads_and_generates_in_fresh_process(
    tmp_path: Path,
    recipe: str,
) -> None:
    fake_root = tmp_path / "fake-backend"
    fake_root.mkdir()
    _write_fake_torchao(fake_root)
    source_root = Path(__file__).parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(fake_root), str(source_root)))
    source = tmp_path / "source.pt"
    output = tmp_path / f"{recipe}.pt"
    export_code = textwrap.dedent(
        f"""
        from pathlib import Path

        import torch

        from tinyllm.config.schema import ModelConfig
        from tinyllm.inference.quantize import quantize_checkpoint
        from tinyllm.model.transformer import TinyLLM

        config = ModelConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=32,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            mlp_ratio=2.0,
            dropout=0.0,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
        )
        torch.save(
            {{
                "version": 2,
                "model_state": TinyLLM(config).state_dict(),
                "config_snapshot": {{"model": config.model_dump(mode="json")}},
                "identity": {{
                    "model_config": config.model_dump(mode="json"),
                    "tokenizer_sha256": "a" * 64,
                    "corpus_sha256": "b" * 64,
                }},
            }},
            Path(r"{source}"),
        )
        quantize_checkpoint(
            Path(r"{source}"),
            Path(r"{output}"),
            "{recipe}",
        )
        """
    )
    exported = subprocess.run(
        [sys.executable, "-c", export_code],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert exported.returncode == 0, exported.stderr

    load_code = textwrap.dedent(
        f"""
        import torch

        from tinyllm.config.schema import GenerationConfig
        from tinyllm.inference.generate import generate
        from tinyllm.inference.quantize import load_quantized_checkpoint

        loaded = load_quantized_checkpoint(r"{output}")
        prompt = torch.tensor([[1, 2]], dtype=torch.long)
        generated = generate(
            loaded.model,
            prompt,
            GenerationConfig(max_new_tokens=1, temperature=0.0, top_p=1.0, seed=7),
        )
        print(loaded.manifest.recipe, list(generated.shape))
        """
    )
    loaded = subprocess.run(
        [sys.executable, "-c", load_code],
        cwd=Path(__file__).parents[2],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert loaded.returncode == 0, loaded.stderr
    assert recipe in loaded.stdout
    assert "[1, 3]" in loaded.stdout
