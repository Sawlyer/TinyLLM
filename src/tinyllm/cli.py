"""Command-line workflows for TinyLLM."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import signal
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch
from tokenizers import Tokenizer

from tinyllm.config.load import load_config
from tinyllm.config.schema import GenerationConfig, ModelConfig, TinyLLMConfig
from tinyllm.data.artifacts import CorpusManifest, load_manifest, sha256_file
from tinyllm.data.audit import audit_artifacts
from tinyllm.data.loader import TokenDataset
from tinyllm.data.prepare import prepare_corpus
from tinyllm.data.validation import validate_manifest_for_model, validate_tokenizer_for_model
from tinyllm.errors import DeviceUnavailableError, TinyLLMUserError
from tinyllm.inference.generate import generate
from tinyllm.inference.quantize import load_quantized_checkpoint, quantize_checkpoint
from tinyllm.model.transformer import TinyLLM, count_parameters_for_config
from tinyllm.training.benchmark import PrecisionBenchmarkConfig, benchmark_precision
from tinyllm.training.checkpoint import (
    ArtifactIdentity,
    TrainingState,
    inspect_checkpoint,
    load_model_checkpoint,
    save_model_checkpoint,
)
from tinyllm.training.checkpoint import (
    load_checkpoint as load_training_checkpoint,
)
from tinyllm.training.checkpoint_manager import (
    list_checkpoint_files,
    plan_checkpoint_retention,
)
from tinyllm.training.evaluation import evaluate_model, write_evaluation_report
from tinyllm.training.metrics import MetricLogger
from tinyllm.training.precision import PrecisionPolicy
from tinyllm.training.run_manifest import finish_run_manifest, start_run_manifest
from tinyllm.training.trainer import Trainer, _resolve_device

DEFAULT_CONFIG = Path("configs/tinyllm.yaml")


def main(arguments: Sequence[str] | None = None) -> None:
    """Run one TinyLLM command."""
    parser = _build_parser()
    args = parser.parse_args(arguments)
    try:
        args.handler(args)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        OSError,
        TinyLLMUserError,
        ValueError,
    ) as error:
        parser.exit(1, f"tinyllm: error: {error}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyllm",
        description="Train and inspect a compact decoder-only language model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="prepare tokenizer and corpus artifacts")
    _add_config_arguments(prepare)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(handler=_prepare_command)

    inspect_data = commands.add_parser(
        "inspect-data", help="validate and summarize prepared corpus artifacts"
    )
    _add_config_arguments(inspect_data)
    inspect_data.set_defaults(handler=_inspect_data_command)

    audit = commands.add_parser(
        "audit-artifacts", help="audit corpus artifacts offline without changing them"
    )
    audit.add_argument("--manifest", type=Path, required=True)
    audit.set_defaults(handler=_audit_artifacts_command)

    train = commands.add_parser("train", help="train or resume a model")
    _add_config_arguments(train)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--resume", type=Path)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--overwrite", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(handler=_train_command)

    evaluate = commands.add_parser("evaluate", help="evaluate a training checkpoint")
    _add_config_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluation_size = evaluate.add_mutually_exclusive_group()
    evaluation_size.add_argument("--batches", type=int, default=8)
    evaluation_size.add_argument("--tokens", type=int)
    evaluate.add_argument("--split", choices=("train", "validation"), default="validation")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.add_argument("--device")
    evaluate.set_defaults(handler=_evaluate_command)

    generate_parser = commands.add_parser("generate", help="generate text from a checkpoint")
    _add_config_arguments(generate_parser, default=None)
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt", action="append", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int)
    generate_parser.add_argument("--temperature", type=float)
    generate_parser.add_argument("--top-k", type=int)
    generate_parser.add_argument("--top-p", type=float)
    generate_parser.add_argument("--repetition-penalty", type=float)
    generate_parser.add_argument("--stop", action="append", default=[])
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--json", action="store_true", dest="json_output")
    generate_parser.add_argument("--device")
    generate_parser.add_argument("--quantized", action="store_true")
    generate_parser.add_argument("--no-cache", action="store_true")
    generate_parser.set_defaults(handler=_generate_command)

    interactive_parser = commands.add_parser(
        "interactive", help="try independent prompts in a local terminal session"
    )
    _add_config_arguments(interactive_parser, default=None)
    interactive_parser.add_argument("--checkpoint", type=Path, required=True)
    interactive_parser.add_argument("--max-new-tokens", type=int)
    interactive_parser.add_argument("--temperature", type=float)
    interactive_parser.add_argument("--top-k", type=int)
    interactive_parser.add_argument("--top-p", type=float)
    interactive_parser.add_argument("--repetition-penalty", type=float)
    interactive_parser.add_argument("--stop", action="append", default=[])
    interactive_parser.add_argument("--seed", type=int)
    interactive_parser.add_argument("--device")
    interactive_parser.add_argument("--quantized", action="store_true")
    interactive_parser.add_argument("--no-cache", action="store_true")
    interactive_parser.add_argument(
        "--history-file",
        type=Path,
        help="append prompts and responses to this JSONL file; disabled by default",
    )
    interactive_parser.set_defaults(handler=_interactive_command)

    profile_parser = commands.add_parser(
        "profile-training", help="profile a short run using real prepared training batches"
    )
    _add_config_arguments(profile_parser)
    profile_parser.add_argument("--warmup-steps", type=int, default=2)
    profile_parser.add_argument("--steps", type=int, default=3, dest="measured_steps")
    profile_parser.add_argument("--device")
    profile_parser.add_argument("--trace", type=Path)
    profile_parser.add_argument("--output", type=Path)
    profile_parser.add_argument("--overwrite", action="store_true")
    profile_parser.set_defaults(handler=_profile_training_command)

    benchmark = commands.add_parser("benchmark-precision", help="compare training precision modes")
    _add_config_arguments(benchmark)
    benchmark.add_argument("--modes", nargs="+", default=["bf16", "fp8"])
    benchmark.add_argument("--batch-size", type=int)
    benchmark.add_argument("--sequence-length", type=int)
    benchmark.add_argument("--warmup-steps", type=int, default=5)
    benchmark.add_argument("--measured-steps", type=int, default=20)
    benchmark.add_argument("--output", type=Path)
    benchmark.add_argument("--overwrite", action="store_true")
    benchmark.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    benchmark.set_defaults(handler=_benchmark_precision_command)

    quantize = commands.add_parser("quantize", help="export a reloadable inference checkpoint")
    _add_config_arguments(quantize)
    quantize.add_argument("--checkpoint", type=Path, required=True)
    quantize.add_argument("--recipe", choices=("bf16", "fp16", "int8", "int4"))
    quantize.add_argument("--output", type=Path)
    quantize.add_argument("--device")
    quantize.add_argument("--overwrite", action="store_true")
    quantize.set_defaults(handler=_quantize_command)

    export_weights = commands.add_parser(
        "export-weights", help="export model-only inference weights from a training checkpoint"
    )
    export_weights.add_argument("--checkpoint", type=Path, required=True)
    export_weights.add_argument("--output", type=Path, required=True)
    export_weights.add_argument("--overwrite", action="store_true")
    export_weights.set_defaults(handler=_export_weights_command)

    checkpoints = commands.add_parser(
        "checkpoints", help="list, inspect, and preview checkpoint retention"
    )
    checkpoint_commands = checkpoints.add_subparsers(dest="checkpoint_command", required=True)

    checkpoint_list = checkpoint_commands.add_parser(
        "list", help="list checkpoint files without loading tensor contents"
    )
    checkpoint_list.add_argument("--directory", type=Path, required=True)
    checkpoint_list.set_defaults(handler=_checkpoints_list_command)

    checkpoint_inspect = checkpoint_commands.add_parser(
        "inspect", help="validate one checkpoint and show its metadata"
    )
    checkpoint_inspect.add_argument("--checkpoint", type=Path, required=True)
    checkpoint_inspect.set_defaults(handler=_checkpoint_inspect_command)

    checkpoint_prune = checkpoint_commands.add_parser(
        "prune", help="preview deterministic checkpoint retention; --apply deletes files"
    )
    checkpoint_prune.add_argument("--directory", type=Path, required=True)
    checkpoint_prune.add_argument("--metrics", type=Path)
    checkpoint_prune.add_argument("--metric", default="validation_loss")
    checkpoint_prune.add_argument("--mode", choices=("min", "max"), default="min")
    checkpoint_prune.add_argument("--run-id")
    checkpoint_prune.add_argument("--keep-last", type=int, default=2)
    checkpoint_prune.add_argument("--keep-every", type=int)
    checkpoint_prune.add_argument("--apply", action="store_true")
    checkpoint_prune.set_defaults(handler=_checkpoint_prune_command)
    return parser


def _checkpoints_list_command(args: argparse.Namespace) -> None:
    files = list_checkpoint_files(args.directory)
    print(
        json.dumps(
            [
                {
                    "path": str(item.path),
                    "step": item.step,
                    "size_bytes": item.size_bytes,
                    "modified_utc": datetime.fromtimestamp(
                        item.modified_time_ns / 1_000_000_000, UTC
                    ).isoformat(),
                }
                for item in files
            ],
            indent=2,
            sort_keys=True,
        )
    )


def _export_weights_command(args: argparse.Namespace) -> None:
    if args.checkpoint.resolve() == args.output.resolve():
        raise ValueError("--output cannot replace the training checkpoint")
    _require_writable_output(
        args.output,
        overwrite=args.overwrite,
        label="inference weights",
    )
    checkpoint = load_model_checkpoint(args.checkpoint, device="cpu")
    save_model_checkpoint(
        args.output,
        checkpoint.model,
        config_snapshot=checkpoint.config_snapshot,
        identity=checkpoint.identity,
        step=checkpoint.step,
        tokens_processed=checkpoint.tokens_processed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "source_checkpoint": str(args.checkpoint),
                "inference_weights": str(args.output),
                "format_version": checkpoint.format_version,
                "step": checkpoint.step,
            },
            sort_keys=True,
        )
    )


def _checkpoint_inspect_command(args: argparse.Namespace) -> None:
    info = inspect_checkpoint(args.checkpoint)
    print(
        json.dumps(
            {
                "path": str(info.path),
                "size_bytes": info.size_bytes,
                "step": info.step,
                "tokens_processed": info.tokens_processed,
                "status": info.status,
                "reason": info.reason,
                "artifact_type": info.artifact_type,
                "format_version": info.format_version,
                "created_at_utc": info.created_at_utc,
                "producer_version": info.producer_version,
                "migrated_from_version": info.migrated_from_version,
                "model_config": info.model_config,
                "tokenizer_sha256": info.tokenizer_sha256,
                "corpus_sha256": info.corpus_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _checkpoint_prune_command(args: argparse.Namespace) -> None:
    metrics_file = args.metrics or args.directory.parent / "metrics.jsonl"
    decisions = plan_checkpoint_retention(
        args.directory,
        metrics_file=metrics_file,
        metric=args.metric,
        mode=args.mode,
        keep_last=args.keep_last,
        keep_every=args.keep_every,
        run_id=args.run_id,
    )
    plan = [
        {
            "action": "keep" if decision.keep else "delete",
            "path": str(decision.path),
            "step": decision.step,
            "size_bytes": decision.size_bytes,
            "resumable": decision.resumable,
            "reason": list(decision.keep_reasons) if decision.keep else decision.inspection_error,
        }
        for decision in decisions
    ]
    print(json.dumps({"apply": args.apply, "plan": plan}, indent=2, sort_keys=True))
    if not args.apply:
        return
    deleted: list[str] = []
    for decision in decisions:
        if decision.keep:
            continue
        if decision.path.is_symlink():
            raise ValueError(f"checkpoint symlinks are not supported: {decision.path}")
        decision.path.unlink()
        deleted.append(str(decision.path))
    print(json.dumps({"deleted": deleted}, indent=2, sort_keys=True))


def _add_config_arguments(
    parser: argparse.ArgumentParser,
    *,
    default: Path | None = DEFAULT_CONFIG,
) -> None:
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one dotted configuration field using a YAML scalar",
    )


def _prepare_command(args: argparse.Namespace) -> None:
    config = _load_args_config(args)
    documents = _local_documents(config.data.dataset_name)
    manifest = prepare_corpus(config, overwrite=args.overwrite, documents=documents)
    print(
        json.dumps(
            {
                "manifest": str(_manifest_path(config)),
                "dataset": manifest.dataset_name,
                "dataset_revision": manifest.dataset_revision,
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "documents": manifest.document_counts,
                "tokens": {
                    split: artifact.token_count for split, artifact in manifest.artifacts.items()
                },
            },
            sort_keys=True,
        )
    )


def _train_command(args: argparse.Namespace) -> None:
    config = _load_args_config(args)
    precision_policy_sink: list[PrecisionPolicy] = []
    report = _build_training_preflight_report(
        config,
        args,
        require_data=not args.dry_run,
        dry_run=args.dry_run,
        precision_policy_sink=precision_policy_sink,
    )
    if args.dry_run:
        print(json.dumps(report, sort_keys=True))
        return
    if report["status"] == "error":
        print(json.dumps(report, sort_keys=True), file=sys.stderr)
        raise ValueError("training preflight failed; see errors in the JSON report")
    checkpoint = args.checkpoint or config.training.checkpoint_dir / "latest.pt"
    manifest_path, manifest = start_run_manifest(config, args, started_at=_utc_timestamp())
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    status = "failed"
    try:
        if precision_policy_sink:
            trainer, _ = _build_cli_trainer(config, precision_policy=precision_policy_sink[0])
        else:
            trainer, _ = _build_cli_trainer(config)
        if isinstance(trainer, Trainer) and trainer.metric_logger is None:
            trainer.metric_logger = MetricLogger(
                config.logging.run_dir,
                run_id=str(manifest["run_id"]),
            )
        if args.resume is not None:
            trainer.load_checkpoint(args.resume)
        summary = trainer.train(max_steps=args.max_steps, overwrite_checkpoints=args.overwrite)
        trainer.save_checkpoint(checkpoint)
        status = "completed"
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "steps": summary.steps,
                    "initial_loss": summary.initial_loss,
                    "final_loss": summary.final_loss,
                    "validation_loss": summary.validation_loss,
                    "tokens_processed": summary.tokens_processed,
                    "elapsed_seconds": summary.elapsed_seconds,
                },
                sort_keys=True,
            )
        )
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        try:
            finish_run_manifest(
                manifest_path,
                manifest,
                config,
                args,
                status=status,
                finished_at=_utc_timestamp(),
            )
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _handle_sigterm(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def _inspect_data_command(args: argparse.Namespace) -> None:
    config = _load_args_config(args)
    manifest = _validated_manifest(config)
    print(
        json.dumps(
            {
                "checksum_validated": True,
                "dataset": manifest.dataset_name,
                "dataset_revision": manifest.dataset_revision,
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "documents": manifest.document_counts,
                "tokenizer_sha256": manifest.tokenizer_sha256,
                "artifacts": {
                    split: {
                        "tokens": artifact.token_count,
                        "sha256": artifact.sha256,
                    }
                    for split, artifact in manifest.artifacts.items()
                },
            },
            sort_keys=True,
        )
    )


def _audit_artifacts_command(args: argparse.Namespace) -> None:
    report = audit_artifacts(args.manifest)
    print(json.dumps(report, sort_keys=True))
    if report["status"] == "error":
        raise TinyLLMUserError("artifact audit found errors")


def _evaluate_command(args: argparse.Namespace) -> None:
    if args.output is not None and args.output.resolve() == args.checkpoint.resolve():
        raise ValueError("--output cannot replace the evaluation checkpoint")
    config = _load_args_config(args)
    if type(args.seed) is not int:
        raise ValueError("--seed must be an integer")
    if args.tokens is not None and (type(args.tokens) is not int or args.tokens <= 0):
        raise ValueError("--tokens must be a positive integer")
    if args.tokens is None and (type(args.batches) is not int or args.batches <= 0):
        raise ValueError("--batches must be a positive integer")
    manifest = _validated_manifest(config, splits={args.split})
    dataset = _dataset(config, manifest, args.split, required=False)
    if dataset is None:
        raise ValueError(f"{args.split} split is too small for configured sequence length")
    device = _resolve_inference_device(args.device or config.training.device)
    loaded = load_model_checkpoint(args.checkpoint, device=device)
    _require_model_config_match(loaded.identity.model_config, config.model)
    report = evaluate_model(
        loaded.model,
        dataset,
        split=args.split,
        seed=args.seed,
        batch_size=config.training.micro_batch_size,
        batches=args.batches if args.tokens is None else None,
        tokens=args.tokens,
        checkpoint_sha256=sha256_file(args.checkpoint),
        corpus_sha256=manifest.artifacts[args.split].sha256,
    )
    if args.output is not None:
        write_evaluation_report(report, args.output, overwrite=args.overwrite)
    print(
        f"Evaluated {report.evaluated_tokens} target tokens across {report.batches} batches "
        f"from split '{report.split}' (seed {report.seed}).",
        file=sys.stderr,
    )
    print(
        f"Mean loss: {report.mean_loss:.6f} nats/token "
        f"(batch standard deviation {report.batch_loss_std:.6f}); "
        f"perplexity: {report.perplexity:.6f}.",
        file=sys.stderr,
    )
    print(f"Checkpoint SHA-256: {report.checkpoint_sha256}", file=sys.stderr)
    print(f"Corpus SHA-256: {report.corpus_sha256}", file=sys.stderr)
    if args.output is not None:
        print(f"Report saved to {args.output}", file=sys.stderr)
    print(
        json.dumps(
            {
                "batches": report.batches,
                "checkpoint": str(args.checkpoint),
                "evaluated_sequences": report.evaluated_sequences,
                "evaluated_tokens": report.evaluated_tokens,
                "mean_loss": report.mean_loss,
                "perplexity": report.perplexity,
                "perplexity_saturated": report.perplexity_saturated,
                "seed": report.seed,
                "split": report.split,
                "validation_loss": report.mean_loss if report.split == "validation" else None,
                "validation_perplexity": report.perplexity
                if report.split == "validation"
                else None,
            },
            sort_keys=True,
        )
    )


def _generate_command(args: argparse.Namespace) -> None:
    if args.quantized:
        if args.config is None:
            raise ValueError("--config is required with --quantized")
        config = load_config(args.config, args.overrides)
    else:
        config = _config_for_checkpoint(args.checkpoint, args.config, args.overrides)
    device = _resolve_inference_device(args.device or config.training.device)
    if args.quantized:
        quantized = load_quantized_checkpoint(args.checkpoint, device=device)
        model = quantized.model
        model_config = quantized.model_config
        expected_tokenizer_sha256 = quantized.manifest.tokenizer_sha256
        if expected_tokenizer_sha256 is None:
            raise ValueError("legacy quantized checkpoint lacks tokenizer identity; re-export it")
    else:
        checkpoint = load_model_checkpoint(args.checkpoint, device=device)
        model = checkpoint.model
        model_config = ModelConfig.model_validate(checkpoint.identity.model_config)
        expected_tokenizer_sha256 = checkpoint.identity.tokenizer_sha256
    _require_model_config_match(model_config.model_dump(mode="json"), config.model)
    tokenizer_path = config.tokenizer.path
    if sha256_file(tokenizer_path) != expected_tokenizer_sha256:
        raise ValueError("tokenizer checksum differs from checkpoint identity")
    tokenizer = _load_tokenizer(tokenizer_path)
    generation_values = config.generation.model_dump()
    generation_values.update(
        {
            field: value
            for field, value in {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "seed": args.seed,
            }.items()
            if value is not None
        }
    )
    if args.stop:
        stop_sequences = [tokenizer.encode(value).ids for value in args.stop]
        if any(not sequence for sequence in stop_sequences):
            raise ValueError("each --stop value must produce at least one token")
        generation_values["stop_token_sequences"] = stop_sequences
    if generation_values["eos_token_id"] is None:
        eos_token_id = tokenizer.token_to_id("<eos>")
        if eos_token_id is None:
            raise ValueError("validated tokenizer does not contain required <eos> token")
        generation_values["eos_token_id"] = eos_token_id
    validate_tokenizer_for_model(tokenizer, model_config)
    generation = GenerationConfig.model_validate(generation_values)
    stop_sequences = [list(sequence) for sequence in generation.stop_token_sequences]
    results = []
    for prompt in args.prompt:
        prompt_ids = tokenizer.encode(prompt).ids
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        if min(prompt_ids) < 0 or max(prompt_ids) >= model_config.vocab_size:
            raise ValueError("prompt contains token IDs outside model vocab_size")
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started_at = perf_counter()
        first_token_at: float | None = None

        def mark_first_token() -> None:
            nonlocal first_token_at
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            first_token_at = perf_counter()

        output_ids = generate(
            model,
            input_ids,
            generation,
            use_cache=not args.no_cache,
            on_first_token=mark_first_token,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = max(perf_counter() - started_at, 1e-12)
        new_ids = output_ids[0, len(prompt_ids) :].tolist()
        matched_stop_length = max(
            (
                len(stop_sequence)
                for stop_sequence in stop_sequences
                if len(new_ids) >= len(stop_sequence)
                and new_ids[-len(stop_sequence) :] == stop_sequence
            ),
            default=0,
        )
        if matched_stop_length:
            del new_ids[-matched_stop_length:]
        continuation = tokenizer.decode(new_ids, skip_special_tokens=False)
        if not continuation:
            continuation = "".join(f"<token:{token_id}>" for token_id in new_ids)
        results.append(
            {
                "prompt": prompt,
                "continuation": continuation,
                "text": f"{prompt}{continuation}",
                "generated_tokens": len(new_ids),
                "tokens_per_second": len(new_ids) / elapsed,
                "first_token_latency_ms": (
                    (first_token_at - started_at) * 1000 if first_token_at is not None else None
                ),
            }
        )
    if args.json_output:
        print(json.dumps({"parameters": generation.model_dump(mode="json"), "results": results}))
    elif len(results) == 1:
        print(results[0]["text"])
    else:
        for index, result in enumerate(results, start=1):
            print(f"[{index}] {result['text']}")


def _interactive_command(args: argparse.Namespace) -> None:
    if args.quantized:
        if args.config is None:
            raise ValueError("--config is required with --quantized")
        config = load_config(args.config, args.overrides)
    else:
        config = _config_for_checkpoint(args.checkpoint, args.config, args.overrides)
    device = _resolve_inference_device(args.device or config.training.device)
    if args.quantized:
        quantized = load_quantized_checkpoint(args.checkpoint, device=device)
        model = quantized.model
        model_config = quantized.model_config
        expected_tokenizer_sha256 = quantized.manifest.tokenizer_sha256
        if expected_tokenizer_sha256 is None:
            raise ValueError("legacy quantized checkpoint lacks tokenizer identity; re-export it")
    else:
        checkpoint = load_model_checkpoint(args.checkpoint, device=device)
        model = checkpoint.model
        model_config = ModelConfig.model_validate(checkpoint.identity.model_config)
        expected_tokenizer_sha256 = checkpoint.identity.tokenizer_sha256
    _require_model_config_match(model_config.model_dump(mode="json"), config.model)
    tokenizer_path = config.tokenizer.path
    if sha256_file(tokenizer_path) != expected_tokenizer_sha256:
        raise ValueError("tokenizer checksum differs from checkpoint identity")
    tokenizer = _load_tokenizer(tokenizer_path)
    generation_values = config.generation.model_dump()
    generation_values.update(
        {
            field: value
            for field, value in {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
                "seed": args.seed,
            }.items()
            if value is not None
        }
    )
    if args.stop:
        stop_sequences = [tokenizer.encode(value).ids for value in args.stop]
        if any(not sequence for sequence in stop_sequences):
            raise ValueError("each --stop value must produce at least one token")
        generation_values["stop_token_sequences"] = stop_sequences
    if generation_values["eos_token_id"] is None:
        eos_token_id = tokenizer.token_to_id("<eos>")
        if eos_token_id is None:
            raise ValueError("validated tokenizer does not contain required <eos> token")
        generation_values["eos_token_id"] = eos_token_id
    validate_tokenizer_for_model(tokenizer, model_config)
    generation = GenerationConfig.model_validate(generation_values)

    from tinyllm.inference.interactive import InteractiveSession

    InteractiveSession(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        generation=generation,
        device=device,
        checkpoint=args.checkpoint,
        use_cache=not args.no_cache,
        history_file=args.history_file,
    ).run()


def _profile_training_command(args: argparse.Namespace) -> None:
    config = _load_args_config(args)
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.measured_steps <= 0:
        raise ValueError("--steps must be positive")
    if args.trace is not None and args.output is not None:
        if args.trace.resolve() == args.output.resolve():
            raise ValueError("--trace and --output must use different files")
    for path, label in ((args.trace, "profiler trace"), (args.output, "profile report")):
        if path is not None and path.exists() and not args.overwrite:
            raise FileExistsError(f"{label} already exists: {path}; pass --overwrite to replace it")

    profile_values = config.model_dump(mode="python")
    total_steps = 1 + args.warmup_steps + args.measured_steps
    training_values = profile_values["training"]
    if args.device is not None:
        training_values["device"] = args.device
    training_values["max_steps"] = total_steps
    training_values["warmup_steps"] = 0
    logging_values = profile_values["logging"]
    with tempfile.TemporaryDirectory(prefix="tinyllm-profile-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        training_values["checkpoint_dir"] = temporary_root / "checkpoints"
        logging_values["run_dir"] = temporary_root / "logs"
        logging_values["log_interval"] = 1
        logging_values["checkpoint_interval"] = total_steps + 1
        profile_config = TinyLLMConfig.model_validate(profile_values)

        setup_started = perf_counter()
        trainer, _ = _build_trainer(profile_config)
        if trainer.device.type == "cuda":
            torch.cuda.synchronize(trainer.device)
        setup_seconds = max(0.0, perf_counter() - setup_started)

        from tinyllm.training.profiling import run_training_profile

        report = run_training_profile(
            trainer,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            trace_path=args.trace,
        )

    device_info: dict[str, object] = {
        "device": str(trainer.device),
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
    }
    if trainer.device.type == "cuda":
        properties = torch.cuda.get_device_properties(trainer.device)
        device_info.update(
            {
                "gpu_name": torch.cuda.get_device_name(trainer.device),
                "gpu_capability": list(torch.cuda.get_device_capability(trainer.device)),
                "gpu_total_memory_bytes": int(properties.total_memory),
            }
        )
    effective_training_config = profile_config.training.model_dump(mode="json")
    effective_training_config["checkpoint_dir"] = "<temporary>"
    report.update(
        {
            "configuration": config.model_dump(mode="json"),
            "effective_training": {
                "device": str(trainer.device),
                "precision": profile_config.training.precision,
                "micro_batch_size": profile_config.training.micro_batch_size,
                "gradient_accumulation_steps": (
                    profile_config.training.gradient_accumulation_steps
                ),
                "sequence_length": profile_config.model.max_seq_len,
                "optimizer_steps": total_steps,
                "unmeasured_priming_steps": 1,
                "training_config": effective_training_config,
            },
            "hardware_and_versions": device_info,
            "measurement_setup_seconds": setup_seconds,
            "measurement_cost_note": (
                "CUDA batch timing synchronizes before and after each profiled transfer; "
                "synchronization and trace export add measurement cost. "
                "profiled_wall_seconds includes profiling, training, and optional trace export. "
                "Model and dataset setup are reported separately."
            ),
        }
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(serialized)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"report": str(args.output), "trace": str(args.trace) if args.trace else None}
        )
    )


def _benchmark_precision_command(args: argparse.Namespace) -> None:
    config = _load_args_config(args)
    output = args.output or config.logging.run_dir / "precision-benchmark.json"
    batch_size = config.training.micro_batch_size if args.batch_size is None else args.batch_size
    sequence_length = (
        config.model.max_seq_len if args.sequence_length is None else args.sequence_length
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if args.measured_steps <= 0:
        raise ValueError("--measured-steps must be positive")
    benchmark_config = PrecisionBenchmarkConfig(
        model=config.model,
        device=config.training.device,
        batch_size=batch_size,
        sequence_length=sequence_length,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        seed=config.training.seed,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        compile=config.training.compile if args.compile is None else args.compile,
        output_path=output,
        overwrite_output=args.overwrite,
    )
    results = benchmark_precision(benchmark_config, args.modes)
    print(json.dumps([result.to_dict() for result in results], sort_keys=True))
    failures = [result for result in results if result.status == "failed"]
    if failures:
        summary = "; ".join(f"{result.mode}: {result.reason}" for result in failures)
        raise ValueError(f"precision benchmark failed: {summary}")


def _quantize_command(args: argparse.Namespace) -> None:
    config = _load_args_config(args)
    recipe = args.recipe or config.quantization.recipe
    if recipe is None:
        raise ValueError("quantization recipe required; pass --recipe")
    canonical_recipe = {
        "int8": "int8-weight-only",
        "int4": "int4-weight-only",
    }.get(recipe, recipe)
    output = args.output or config.quantization.output_dir / (f"{args.checkpoint.stem}-{recipe}.pt")
    manifest = quantize_checkpoint(
        args.checkpoint,
        output,
        canonical_recipe,
        overwrite=args.overwrite,
        device=args.device or config.training.device,
    )
    print(json.dumps({"output": str(output), "manifest": manifest.to_dict()}, sort_keys=True))


def _build_training_preflight_report(
    config: TinyLLMConfig,
    args: argparse.Namespace | None = None,
    *,
    require_data: bool,
    dry_run: bool = False,
    precision_policy_sink: list[PrecisionPolicy] | None = None,
) -> dict[str, object]:
    args = args or argparse.Namespace(checkpoint=None, resume=None, overwrite=False, max_steps=None)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    fp8_requested = False

    def issue(
        target: list[dict[str, str]], field: str, path: Path | None, cause: str, fix: str
    ) -> None:
        target.append(
            {
                "field": field,
                "path": str(path) if path is not None else "",
                "cause": cause,
                "fix": fix,
            }
        )

    try:
        device = _resolve_device(config.training.device)
        if device.type == "cuda" and device.index is not None:
            count = torch.cuda.device_count()
            if device.index >= count:
                raise RuntimeError(f"{device} requested but only {count} available CUDA device(s)")
    except (RuntimeError, ValueError, TypeError) as error:
        device = None
        issue(
            errors,
            "training.device",
            None,
            str(error),
            "Select an available device such as cpu, cuda, or cuda:N.",
        )

    if device is not None:
        precision = config.training.precision.lower()
        try:
            if precision == "fp8":
                fp8_requested = True
                if device.type != "cuda" or not torch.cuda.is_available():
                    raise ValueError("FP8 requires an available CUDA device")
                if device.index is not None and device.index >= torch.cuda.device_count():
                    raise ValueError(f"FP8 CUDA device index {device.index} is unavailable")
                capability = torch.cuda.get_device_capability(device)
                if capability < (8, 9):
                    raise ValueError(
                        f"FP8 requires CUDA compute capability 8.9+, found {capability}"
                    )
                if not hasattr(torch, "compile"):
                    raise ValueError("FP8 requires torch.compile")
                if importlib.util.find_spec("torchao") is None:
                    raise ValueError("FP8 requires the optional torchao package")
                if dry_run:
                    issue(
                        warnings,
                        "training.precision",
                        None,
                        "FP8 kernel probe deferred; dry-run does not compile kernels",
                        "Run training on target GPU to validate FP8 kernel and TorchAO conversion.",
                    )
            else:
                PrecisionPolicy.create(precision, device)
        except (RuntimeError, ValueError, ImportError, TinyLLMUserError) as error:
            issue(
                errors,
                "training.precision",
                None,
                str(error),
                "Choose a supported precision for the selected device.",
            )

    manifest_path = _manifest_path(config)
    manifest: CorpusManifest | None = None
    train_dataset: TokenDataset | None = None
    if manifest_path.is_file():
        try:
            manifest = _validated_manifest(config)
            train_dataset = _dataset(config, manifest, "train", required=True)
            _dataset(config, manifest, "validation", required=False)
        except (OSError, ValueError, KeyError) as error:
            issue(
                errors,
                "data.manifest",
                manifest_path,
                str(error),
                "Run `tinyllm inspect-data` and prepare compatible corpus artifacts.",
            )
            train_dataset = None
    elif require_data:
        issue(
            errors,
            "data.manifest",
            manifest_path,
            "prepared corpus manifest is missing",
            "Run `tinyllm prepare` with this configuration before training.",
        )
    else:
        issue(
            warnings,
            "data.manifest",
            manifest_path,
            "prepared corpus is absent; dry-run will not download or create it",
            "Run `tinyllm prepare` before training.",
        )

    checkpoint = args.checkpoint or config.training.checkpoint_dir / "latest.pt"
    requested_steps = getattr(args, "max_steps", None)
    steps = config.training.max_steps if requested_steps is None else requested_steps
    if type(steps) is not int or steps <= 0 or steps > config.training.max_steps:
        issue(
            errors,
            "training.max_steps",
            None,
            f"requested steps {steps!r} must be between 1 and {config.training.max_steps}",
            "Choose --max-steps within the configured training limit.",
        )
    if args.resume is not None and not Path(args.resume).is_file():
        issue(
            errors,
            "resume",
            Path(args.resume),
            "resume checkpoint does not exist",
            "Provide an existing compatible checkpoint or omit --resume.",
        )
    elif (
        args.resume is not None
        and train_dataset is not None
        and manifest is not None
        and not errors
    ):
        try:
            train_artifact = manifest.artifacts["train"]
            expected_identity = ArtifactIdentity(
                model_config=config.model.model_dump(mode="json"),
                tokenizer_sha256=manifest.tokenizer_sha256,
                corpus_sha256=train_artifact.sha256,
            )
            state = load_training_checkpoint(Path(args.resume), expected_identity)
            _validate_resume_preflight(
                config,
                train_dataset,
                state,
                max_steps=steps,
                device=device,
            )
        except (OSError, ValueError, TypeError, TinyLLMUserError) as error:
            issue(
                errors,
                "resume",
                Path(args.resume),
                str(error),
                "Use a readable checkpoint matching model, data, tokenizer, and training settings.",
            )
    if checkpoint.exists() and not args.overwrite:
        issue(
            errors,
            "checkpoint",
            checkpoint,
            "output checkpoint already exists",
            "Choose another --checkpoint path or pass --overwrite.",
        )

    parameters = count_parameters_for_config(config.model)
    checkpoint_bytes = parameters * 12 + 16 * 1024 * 1024
    interval_count = steps // config.logging.checkpoint_interval
    checkpoint_storage = checkpoint_bytes * (interval_count + 3)
    estimated_bytes = checkpoint_storage + checkpoint_bytes
    if not args.overwrite:
        for step in range(
            config.logging.checkpoint_interval, steps + 1, config.logging.checkpoint_interval
        ):
            periodic_checkpoint = config.training.checkpoint_dir / f"step-{step:08d}.pt"
            if periodic_checkpoint.exists():
                issue(
                    errors,
                    "checkpoint",
                    periodic_checkpoint,
                    "periodic checkpoint already exists",
                    "Choose --overwrite or remove the conflicting checkpoint file.",
                )
    output_roots = [checkpoint.parent, config.training.checkpoint_dir, config.logging.run_dir]
    for output_root in dict.fromkeys(output_roots):
        probe = output_root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        if not probe.is_dir():
            issue(
                errors,
                "output_directory",
                output_root,
                "an ancestor of the output directory is not a directory",
                "Choose a writable directory path.",
            )
        elif not os.access(probe, os.W_OK):
            issue(
                errors,
                "output_directory",
                output_root,
                f"directory is not writable: {probe}",
                "Grant write access or choose another output directory.",
            )

    storage_targets = [
        (config.training.checkpoint_dir, checkpoint_storage),
        (checkpoint.parent, checkpoint_bytes),
    ]
    volumes: dict[str, tuple[Path, int]] = {}
    for target_path, required_bytes in storage_targets:
        probe = target_path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        volume = str(probe.anchor).casefold()
        prior_path, prior_required = volumes.get(volume, (probe, 0))
        volumes[volume] = (prior_path, prior_required + required_bytes)
    for probe, required_bytes in volumes.values():
        try:
            free_bytes = shutil.disk_usage(probe).free
            if free_bytes < required_bytes:
                issue(
                    errors,
                    "storage",
                    probe,
                    "estimated checkpoint storage is "
                    f"{required_bytes} bytes, only {free_bytes} bytes are free",
                    "Free disk space or reduce training steps/checkpoint frequency.",
                )
        except OSError as error:
            issue(
                warnings,
                "storage",
                probe,
                f"could not inspect free space: {error}",
                "Confirm sufficient disk space on the target volume.",
            )

    if fp8_requested and not dry_run and not errors:
        try:
            fp8_policy = PrecisionPolicy.create("fp8", device)
            if precision_policy_sink is not None:
                precision_policy_sink.append(fp8_policy)
        except (RuntimeError, ValueError, ImportError, TinyLLMUserError) as error:
            issue(
                errors,
                "training.precision",
                None,
                str(error),
                "Choose a supported precision for the selected device.",
            )

    status = "error" if errors else "warning" if warnings else "ok"
    return {
        "status": status,
        "data": {
            "manifest": str(manifest_path),
            "available": manifest is not None,
            "dataset": None if manifest is None else manifest.dataset_name,
            "revision": None if manifest is None else manifest.dataset_revision,
        },
        "model": {"parameter_count": parameters},
        "training": {
            "device": str(device) if device is not None else config.training.device,
            "precision": config.training.precision,
            "steps": steps,
            "checkpoint_bytes_estimate": estimated_bytes,
        },
        "outputs": {"checkpoint": str(checkpoint), "run_dir": str(config.logging.run_dir)},
        "errors": errors,
        "warnings": warnings,
    }


def _validate_resume_preflight(
    config: TinyLLMConfig,
    train_dataset: TokenDataset,
    state: TrainingState,
    *,
    max_steps: int,
    device: torch.device,
) -> None:
    snapshot = state.config_snapshot
    model_snapshot = snapshot.get("model")
    training_snapshot = snapshot.get("training")
    if not isinstance(model_snapshot, Mapping) or not isinstance(training_snapshot, Mapping):
        raise ValueError("checkpoint configuration snapshot is invalid")
    if dict(model_snapshot) != config.model.model_dump(mode="json"):
        raise ValueError("checkpoint model configuration is incompatible")
    fields = (
        "learning_rate",
        "min_learning_rate",
        "warmup_steps",
        "max_steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "precision",
        "weight_decay",
        "max_grad_norm",
        "compile",
    )
    current_training = config.training.model_dump(mode="json")
    for field in fields:
        if training_snapshot.get(field) != current_training[field]:
            raise ValueError(f"checkpoint training {field} is incompatible")

    scheduler = dict(state.scheduler_state)
    if set(scheduler) != {"last_step", "total_steps"} or scheduler.get("last_step") != state.step:
        raise ValueError("checkpoint scheduler state is inconsistent")
    if scheduler.get("total_steps") != config.training.max_steps:
        raise ValueError("checkpoint scheduler configuration is incompatible")
    if state.step > config.training.max_steps:
        raise ValueError("checkpoint step exceeds configured total steps")
    if max_steps <= state.step:
        raise ValueError("max_steps must be greater than the current checkpoint step")
    tokens_per_step = (
        config.training.micro_batch_size
        * train_dataset.sequence_length
        * config.training.gradient_accumulation_steps
    )
    if state.tokens_processed != state.step * tokens_per_step:
        raise ValueError("checkpoint token count is inconsistent with step")
    train_dataset.validate_state_dict(state.loader_rng_state)
    random.Random().setstate(state.python_rng_state)
    torch.Generator(device="cpu").set_state(state.torch_rng_state)

    metadata = dict(state.cuda_rng_metadata)
    required = {
        "available",
        "device_count",
        "active_device",
        "training_device_type",
        "training_device_index",
    }
    if set(metadata) != required:
        raise ValueError("checkpoint CUDA RNG metadata has invalid fields")
    available = metadata["available"]
    device_count = metadata["device_count"]
    active_device = metadata["active_device"]
    if type(available) is not bool or type(device_count) is not int:
        raise ValueError("checkpoint CUDA RNG metadata is invalid")
    if available:
        if (
            device_count <= 0
            or type(active_device) is not int
            or not 0 <= active_device < device_count
            or len(state.cuda_rng_states) != device_count
        ):
            raise ValueError("checkpoint CUDA RNG metadata is inconsistent")
    elif device_count != 0 or active_device is not None or state.cuda_rng_states:
        raise ValueError("checkpoint CUDA RNG metadata is inconsistent")
    for cuda_state in state.cuda_rng_states:
        if (
            cuda_state.device.type != "cpu"
            or cuda_state.dtype != torch.uint8
            or cuda_state.numel() == 0
        ):
            raise ValueError("checkpoint CUDA RNG state is invalid")
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    active_device = torch.cuda.current_device() if cuda_available else None
    training_device_index = device.index
    if device.type == "cuda" and training_device_index is None:
        training_device_index = active_device
    current_cuda_metadata = {
        "available": cuda_available,
        "device_count": device_count,
        "active_device": active_device,
        "training_device_type": device.type,
        "training_device_index": training_device_index,
    }
    for field, current_value in current_cuda_metadata.items():
        if metadata[field] != current_value:
            raise ValueError(f"checkpoint CUDA/device metadata {field} is incompatible")
    for device_index, cuda_state in enumerate(state.cuda_rng_states):
        try:
            torch.Generator(device=f"cuda:{device_index}").set_state(cuda_state)
        except RuntimeError as error:
            raise ValueError("checkpoint CUDA RNG state is invalid") from error


def _safe_perplexity(loss: float) -> float:
    return math.exp(min(loss, 80.0))


def _load_args_config(args: argparse.Namespace) -> TinyLLMConfig:
    if args.config is None:
        raise ValueError("--config is required for this command")
    return load_config(args.config, args.overrides)


def _config_for_checkpoint(
    checkpoint: Path,
    config_path: Path | None,
    overrides: list[str],
) -> TinyLLMConfig:
    if config_path is not None:
        return load_config(config_path, overrides)
    if overrides:
        raise ValueError("--set requires --config when loading checkpoint configuration")
    loaded = load_model_checkpoint(checkpoint, device="cpu")
    snapshot = loaded.config_snapshot
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise ValueError("checkpoint does not contain a configuration snapshot")
    return TinyLLMConfig.model_validate(dict(snapshot))


def _build_trainer(
    config: TinyLLMConfig,
    *,
    precision_policy: PrecisionPolicy | None = None,
) -> tuple[Trainer, TokenDataset | None]:
    manifest = _validated_manifest(config)
    resolved_device = _resolve_device(config.training.device)
    resolved_precision = precision_policy or PrecisionPolicy.create(
        config.training.precision, resolved_device
    )
    train_dataset = _dataset(config, manifest, "train", required=True)
    validation_dataset = _dataset(config, manifest, "validation", required=False)
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    model = TinyLLM(config.model)
    trainer = Trainer(
        config,
        model,
        train_dataset,
        validation_dataset,
        precision_policy=resolved_precision,
    )
    return trainer, validation_dataset


def _build_cli_trainer(
    config: TinyLLMConfig,
    *,
    precision_policy: PrecisionPolicy | None = None,
) -> tuple[Trainer, TokenDataset | None]:
    try:
        return _build_trainer(config, precision_policy=precision_policy)
    except RuntimeError as error:
        message = str(error)
        if message.startswith(("CUDA training requested", "MPS training requested")):
            raise ValueError(message) from error
        raise


def _load_tokenizer(path: Path) -> Tokenizer:
    try:
        return Tokenizer.from_file(str(path))
    except Exception as error:
        raise ValueError(f"invalid tokenizer file: {path}") from error


def _dataset(
    config: TinyLLMConfig,
    manifest: CorpusManifest,
    split: str,
    *,
    required: bool,
) -> TokenDataset | None:
    artifact = manifest.artifacts[split]
    if artifact.token_count < config.model.max_seq_len + 1:
        if required:
            raise ValueError(
                f"{split} split has {artifact.token_count} tokens; "
                f"need at least {config.model.max_seq_len + 1}"
            )
        return None
    return TokenDataset(
        manifest,
        split,
        config.model.max_seq_len,
        config.training.seed,
    )


def _manifest_path(config: TinyLLMConfig) -> Path:
    return config.data.output_dir / "manifest.json"


def _validated_manifest(
    config: TinyLLMConfig,
    *,
    splits: Iterable[str] | None = None,
) -> CorpusManifest:
    manifest = load_manifest(_manifest_path(config), splits=splits)
    if manifest.data_config != config.data.model_dump(mode="json"):
        raise ValueError("corpus manifest data configuration is incompatible with config")
    if manifest.tokenizer_config != config.tokenizer.model_dump(mode="json"):
        raise ValueError("corpus manifest tokenizer configuration is incompatible with config")
    if manifest.dataset_name != config.data.dataset_name:
        raise ValueError("corpus manifest dataset name is incompatible with config")
    if manifest.dataset_requested_revision != config.data.dataset_revision:
        raise ValueError("corpus manifest dataset revision is incompatible with config")
    if _resolved_tokenizer_path(config, manifest).resolve() != config.tokenizer.path.resolve():
        raise ValueError("corpus manifest tokenizer path is incompatible with config")
    validate_manifest_for_model(manifest, config.model, splits=splits)
    return manifest


def _require_model_config_match(
    checkpoint_model_config: Mapping[str, object],
    configured_model: ModelConfig,
) -> None:
    if dict(checkpoint_model_config) != configured_model.model_dump(mode="json"):
        raise ValueError("checkpoint model configuration is incompatible with config")


def _resolve_inference_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid inference device: {name!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError("CUDA inference requested but CUDA is unavailable")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise DeviceUnavailableError("MPS inference requested but MPS is unavailable")
    return device


def _resolved_tokenizer_path(config: TinyLLMConfig, manifest: CorpusManifest) -> Path:
    path = Path(manifest.tokenizer_path)
    if not path.is_absolute():
        path = _manifest_path(config).parent / path
    return path


def _local_documents(dataset_name: str) -> Iterable[str] | None:
    source = Path(dataset_name)
    if not source.is_file():
        return None

    def read_lines() -> Iterable[str]:
        with source.open(encoding="utf-8") as source_file:
            for line in source_file:
                text = line.strip()
                if text:
                    yield text

    return read_lines()


def _require_writable_output(path: Path, *, overwrite: bool, label: str) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{label} already exists: {path}; pass --overwrite to replace it")

