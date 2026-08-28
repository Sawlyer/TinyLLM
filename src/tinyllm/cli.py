"""Commandes CLI de TinyLLM."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
from tokenizers import Tokenizer

from tinyllm.config.load import load_config
from tinyllm.config.schema import GenerationConfig, ModelConfig, TinyLLMConfig
from tinyllm.data.artifacts import CorpusManifest, load_manifest, sha256_file
from tinyllm.data.loader import TokenDataset
from tinyllm.data.prepare import prepare_corpus
from tinyllm.data.validation import validate_manifest_for_model, validate_tokenizer_for_model
from tinyllm.errors import DeviceUnavailableError, TinyLLMUserError
from tinyllm.inference.generate import generate
from tinyllm.inference.quantize import load_quantized_checkpoint, quantize_checkpoint
from tinyllm.model.transformer import TinyLLM, count_parameters
from tinyllm.training.benchmark import PrecisionBenchmarkConfig, benchmark_precision
from tinyllm.training.checkpoint import load_model_checkpoint
from tinyllm.training.trainer import Trainer

DEFAULT_CONFIG = Path("configs/tinyllm.yaml")


def main(arguments: Sequence[str] | None = None) -> None:
    """Exécuter une commande TinyLLM."""
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
    evaluate.add_argument("--batches", type=int, default=8)
    evaluate.add_argument("--device")
    evaluate.set_defaults(handler=_evaluate_command)

    generate_parser = commands.add_parser("generate", help="generate text from a checkpoint")
    _add_config_arguments(generate_parser, default=None)
    generate_parser.add_argument("--checkpoint", type=Path, required=True)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int)
    generate_parser.add_argument("--temperature", type=float)
    generate_parser.add_argument("--top-k", type=int)
    generate_parser.add_argument("--top-p", type=float)
    generate_parser.add_argument("--seed", type=int)
    generate_parser.add_argument("--device")
    generate_parser.add_argument("--quantized", action="store_true")
    generate_parser.add_argument("--no-cache", action="store_true")
    generate_parser.set_defaults(handler=_generate_command)

    benchmark = commands.add_parser("benchmark-precision", help="compare training precision modes")
    _add_config_arguments(benchmark)
    benchmark.add_argument("--modes", nargs="+", default=["bf16"])
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
    quantize.add_argument("--recipe", choices=("bf16", "fp16", "int8"))
    quantize.add_argument("--output", type=Path)
    quantize.add_argument("--device")
    quantize.add_argument("--overwrite", action="store_true")
    quantize.set_defaults(handler=_quantize_command)
    return parser


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
    if args.dry_run:
        _print_train_dry_run(config)
        return
    checkpoint = args.checkpoint or config.training.checkpoint_dir / "latest.pt"
    _require_writable_output(checkpoint, overwrite=args.overwrite, label="checkpoint")
    trainer, _ = _build_cli_trainer(config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    summary = trainer.train(max_steps=args.max_steps, overwrite_checkpoints=args.overwrite)
    trainer.save_checkpoint(checkpoint)
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


def _evaluate_command(args: argparse.Namespace) -> None:
    config = _load_args_config(args)
    if type(args.batches) is not int or args.batches <= 0:
        raise ValueError("--batches must be a positive integer")
    manifest = _validated_manifest(config, splits={"validation"})
    validation_dataset = _dataset(config, manifest, "validation", required=False)
    if validation_dataset is None:
        raise ValueError("validation split is too small for configured sequence length")
    device = _resolve_inference_device(args.device or config.training.device)
    loaded = load_model_checkpoint(args.checkpoint, device=device)
    _require_model_config_match(loaded.identity.model_config, config.model)
    validation_loss = _evaluate_validation_model(
        loaded.model,
        validation_dataset,
        args.batches,
        batch_size=config.training.micro_batch_size,
        device=device,
    )
    print(
        json.dumps(
            {
                "batches": args.batches,
                "checkpoint": str(args.checkpoint),
                "validation_loss": validation_loss,
                "validation_perplexity": _safe_perplexity(validation_loss),
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
    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        raise ValueError("prompt produced no tokens")
    generation_values = config.generation.model_dump()
    generation_values.update(
        {
            field: value
            for field, value in {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "seed": args.seed,
            }.items()
            if value is not None
        }
    )
    if generation_values["eos_token_id"] is None:
        eos_token_id = tokenizer.token_to_id("<eos>")
        if eos_token_id is None:
            raise ValueError("validated tokenizer does not contain required <eos> token")
        generation_values["eos_token_id"] = eos_token_id
    validate_tokenizer_for_model(tokenizer, model_config)
    if min(prompt_ids) < 0 or max(prompt_ids) >= model_config.vocab_size:
        raise ValueError("prompt contains token IDs outside model vocab_size")
    generation = GenerationConfig.model_validate(generation_values)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = generate(
        model,
        input_ids,
        generation,
        use_cache=not args.no_cache,
    )
    new_ids = output_ids[0, len(prompt_ids) :].tolist()
    continuation = tokenizer.decode(new_ids, skip_special_tokens=False)
    if not continuation:
        continuation = "".join(f"<token:{token_id}>" for token_id in new_ids)
    print(f"{args.prompt}{continuation}")


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


def _print_train_dry_run(config: TinyLLMConfig) -> None:
    manifest_path = _manifest_path(config)
    expected_steps = config.training.max_steps
    if manifest_path.is_file():
        manifest = _validated_manifest(config)
        dataset_identity = (
            f"validated manifest {manifest.dataset_name}@{manifest.dataset_revision} "
            f"fingerprint={manifest.dataset_fingerprint} "
            f"tokenizer_sha256={manifest.tokenizer_sha256} "
            f"train_sha256={manifest.artifacts['train'].sha256}"
        )
    else:
        requested_revision = config.data.dataset_revision or "latest resolved by prepare"
        dataset_identity = (
            f"validated config {config.data.dataset_name}@{requested_revision}; "
            "prepared artifacts not present"
        )
    model = TinyLLM(config.model)
    total_parameters, trainable_parameters = count_parameters(model)
    effective_sequences = (
        config.training.micro_batch_size * config.training.gradient_accumulation_steps
    )
    effective_tokens = effective_sequences * config.model.max_seq_len
    if manifest_path.is_file():
        total_documents = sum(manifest.document_counts.values())
        expected_steps = math.ceil(
            total_documents * (1.0 - config.data.validation_fraction) / effective_sequences
        )
    device = config.training.device
    if device == "cuda":
        device = "cuda (RTX 5080 target; availability not probed during dry-run)"
    print(f"dataset identity: {dataset_identity}")
    print(
        "parameters: "
        f"{total_parameters:,} total ({total_parameters / 1_000_000:.2f}M), "
        f"{trainable_parameters:,} trainable"
    )
    print(
        f"effective batch: {effective_sequences} sequences, "
        f"{effective_tokens:,} tokens per optimizer step"
    )
    print(f"expected steps: {expected_steps}")
    print(f"precision: {config.training.precision.upper()}")
    print(f"device: {device}")


def _evaluate_validation_model(
    model: torch.nn.Module,
    dataset: TokenDataset,
    batches: int,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_index in range(batches):
            start = batch_index * batch_size
            positions = torch.arange(
                start,
                start + batch_size,
                dtype=torch.long,
            ) % len(dataset)
            inputs, targets = dataset.batch(positions)
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs, targets)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise ValueError("validation produced a non-finite loss")
            losses.append(float(output.loss))
    return sum(losses) / len(losses)


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


def _build_trainer(config: TinyLLMConfig) -> tuple[Trainer, TokenDataset | None]:
    manifest = _validated_manifest(config)
    train_dataset = _dataset(config, manifest, "train", required=True)
    validation_dataset = _dataset(config, manifest, "validation", required=False)
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    model = TinyLLM(config.model)
    trainer = Trainer(config, model, train_dataset, validation_dataset)
    return trainer, validation_dataset


def _build_cli_trainer(config: TinyLLMConfig) -> tuple[Trainer, TokenDataset | None]:
    try:
        return _build_trainer(config)
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
