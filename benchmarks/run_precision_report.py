from __future__ import annotations

import json
import platform
import statistics
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import torch

from tinyllm.cli import _dataset, _validated_manifest
from tinyllm.config.load import load_config
from tinyllm.data.artifacts import sha256_file
from tinyllm.inference.precision_report import benchmark_inference
from tinyllm.inference.quantize import load_quantized_checkpoint
from tinyllm.training.checkpoint import load_model_checkpoint

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "tinyllm.yaml"
CHECKPOINT = ROOT / "checkpoints" / "resumed.pt"
INT8_CHECKPOINT = ROOT / "exports" / "tinyllm-int8.pt"
OUTPUT = ROOT / "docs" / "benchmarks" / "precision-report.json"
TRAINING_RUNS = tuple(ROOT / "runs" / f"precision-comparison-{index}.json" for index in range(1, 4))
MODES = ("fp32", "fp16", "bf16", "int8")


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the precision report")
    config = load_config(CONFIG, [])
    manifest = _validated_manifest(config, splits={"validation"})
    dataset = _dataset(config, manifest, "validation", required=True)
    assert dataset is not None
    batches = _fixed_batches(dataset, config.training.micro_batch_size, count=8)
    reference_model = load_model_checkpoint(CHECKPOINT, device=device).model
    reference_logits = _reference_logits(reference_model, batches, device)
    del reference_model
    torch.cuda.empty_cache()

    inference_results = []
    for mode in MODES:
        if mode == "int8":
            model = load_quantized_checkpoint(INT8_CHECKPOINT, device=device).model
            autocast_context = nullcontext
        else:
            model = load_model_checkpoint(CHECKPOINT, device=device).model
            autocast_context = _autocast(mode, device)
        result = benchmark_inference(
            model,
            batches,
            mode=mode,
            device=device,
            autocast_context=autocast_context,
            reference_logits=reference_logits,
            repetitions=5,
            warmup_repetitions=3,
        )
        inference_results.append(
            {
                "mode": result.mode,
                "mean_loss": result.mean_loss,
                "perplexity": result.perplexity,
                "tokens_per_second_runs": list(result.tokens_per_second),
                "tokens_per_second_mean": statistics.mean(result.tokens_per_second),
                "tokens_per_second_median": statistics.median(result.tokens_per_second),
                "tokens_per_second_stdev": statistics.stdev(result.tokens_per_second),
                "peak_allocated_bytes": result.peak_allocated_bytes,
                "mean_absolute_logit_error_vs_fp32": result.mean_absolute_logit_error,
                "max_absolute_logit_error_vs_fp32": result.max_absolute_logit_error,
            }
        )
        del model
        torch.cuda.empty_cache()

    payload = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchao": version("torchao"),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "dataset_revision": manifest.dataset_revision,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "checkpoint_step": 210,
            "validation_batches": 8,
            "batch_size": config.training.micro_batch_size,
            "sequence_length": config.model.max_seq_len,
        },
        "training": _aggregate_training_runs(),
        "inference": inference_results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


def _fixed_batches(dataset, batch_size: int, *, count: int):
    batches = []
    for batch_index in range(count):
        start = batch_index * batch_size
        positions = torch.arange(start, start + batch_size, dtype=torch.long) % len(dataset)
        batches.append(dataset.batch(positions))
    return batches


def _reference_logits(model, batches, device: torch.device):
    model.eval()
    logits = []
    with torch.inference_mode():
        for inputs, targets in batches:
            logits.append(model(inputs.to(device), targets.to(device)).logits.float().cpu())
    return logits


def _autocast(mode: str, device: torch.device):
    if mode == "fp32":
        return nullcontext
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[mode]
    return lambda: torch.autocast(device_type=device.type, dtype=dtype)


def _aggregate_training_runs():
    rows = {}
    for path in TRAINING_RUNS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            rows.setdefault(result["mode"], []).append(result)
    return [
        {
            "mode": mode,
            "tokens_per_second_runs": [row["tokens_per_second"] for row in mode_rows],
            "tokens_per_second_mean": statistics.mean(
                row["tokens_per_second"] for row in mode_rows
            ),
            "tokens_per_second_stdev": statistics.stdev(
                row["tokens_per_second"] for row in mode_rows
            ),
            "peak_allocated_bytes": mode_rows[0]["peak_allocated_bytes"],
            "mean_loss": statistics.mean(row["mean_loss"] for row in mode_rows),
        }
        for mode, mode_rows in sorted(rows.items())
    ]


if __name__ == "__main__":
    main()
