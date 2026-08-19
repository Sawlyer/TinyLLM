# TinyLLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ~30M-parameter decoder-only Transformer that trains from scratch on FineWeb-Edu 100M and runs efficiently on one RTX 5080.

**Architecture:** A typed configuration drives independent data, model, training, and inference packages. Thin CLI commands compose tested APIs; binary corpus artifacts and checkpoints carry identities that prevent incompatible resume or evaluation.

**Tech Stack:** Python 3.12, PyTorch CUDA, TorchAO optional, Hugging Face datasets/tokenizers, Pydantic, PyYAML, TensorBoard, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-19-tinyllm-design.md`

## Global Constraints

- Default target is one NVIDIA GeForce RTX 5080 with 16 GB VRAM.
- Default model is 8 layers, width 512, 8 query heads, 4 KV heads, vocabulary 16,384, context 512.
- BF16 is training default; FP8 is explicit and experimental; unsupported modes fail rather than silently fall back.
- Dataset source is `codelion/fineweb-edu-100M`; tokenizer trains only on training split.
- Dataset binaries, checkpoints, logs, and generated exports are excluded from Git.
- CPU tests never require full dataset or CUDA; CUDA tests use `pytest.mark.cuda`.
- Every production behavior follows red-green-refactor.

---

### Task 1: Project Foundation and Typed Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `configs/tinyllm.yaml`
- Create: `src/tinyllm/__init__.py`
- Create: `src/tinyllm/config/schema.py`
- Create: `src/tinyllm/config/load.py`
- Create: `src/tinyllm/cli.py`
- Test: `tests/config/test_config.py`

**Interfaces:**
- Produces: `TinyLLMConfig`, `load_config(path: Path, overrides: list[str]) -> TinyLLMConfig`, `main() -> None`.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_default_config_has_valid_gqa_shape(config_path):
    cfg = load_config(config_path, [])
    assert cfg.model.n_heads == 8
    assert cfg.model.n_kv_heads == 4
    assert cfg.model.d_model % cfg.model.n_heads == 0

def test_override_changes_nested_value(config_path):
    cfg = load_config(config_path, ["training.precision=fp32"])
    assert cfg.training.precision == "fp32"

def test_invalid_head_ratio_is_rejected(config_path):
    with pytest.raises(ValueError, match="n_heads.*n_kv_heads"):
        load_config(config_path, ["model.n_kv_heads=3"])
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/config/test_config.py -v`
Expected: collection fails because `tinyllm.config` does not exist.

- [ ] **Step 3: Add packaging, dependencies, schema, loader, defaults, CLI shell, and ignored runtime paths**

Use Pydantic models `DataConfig`, `TokenizerConfig`, `ModelConfig`, `TrainingConfig`, `LoggingConfig`, `GenerationConfig`, `QuantizationConfig`, composed by `TinyLLMConfig`. Parse overrides as dotted YAML scalars and reject unknown keys. Register console script `tinyllm = "tinyllm.cli:main"`.

- [ ] **Step 4: Verify GREEN**

Run: `uv sync --extra dev` then `uv run pytest tests/config/test_config.py -v`
Expected: all configuration tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore configs src tests
git commit -m "feat: establish TinyLLM configuration"
```

### Task 2: Deterministic Corpus Preparation and BPE Tokenizer

**Files:**
- Create: `src/tinyllm/data/clean.py`
- Create: `src/tinyllm/data/split.py`
- Create: `src/tinyllm/data/tokenizer.py`
- Create: `src/tinyllm/data/artifacts.py`
- Create: `src/tinyllm/data/prepare.py`
- Test: `tests/data/test_prepare.py`
- Test: `tests/data/test_artifacts.py`

**Interfaces:**
- Consumes: `DataConfig`, `TokenizerConfig`.
- Produces: `normalize_document(text: str) -> str | None`, `split_for_id(document_id: str, seed: int, validation_fraction: float) -> str`, `train_tokenizer(texts: Iterable[str], config: TokenizerConfig, output: Path) -> Tokenizer`, `prepare_corpus(config: TinyLLMConfig, overwrite: bool = False) -> CorpusManifest`, `load_manifest(path: Path) -> CorpusManifest`.

- [ ] **Step 1: Write failing deterministic preparation tests**

```python
def test_cleaning_normalizes_nfc_and_drops_blank():
    assert normalize_document("  cafe\u0301  ") == "café"
    assert normalize_document(" \n\t ") is None

def test_split_is_stable():
    first = split_for_id("doc-42", seed=1337, validation_fraction=0.01)
    assert first == split_for_id("doc-42", seed=1337, validation_fraction=0.01)

def test_binary_artifact_round_trip(tmp_path):
    tokens = np.array([1, 8, 3, 2], dtype=np.uint16)
    manifest = write_token_artifact(tmp_path, "train", tokens, tokenizer_sha256="abc")
    loaded = open_token_artifact(tmp_path, manifest)
    assert loaded.tolist() == tokens.tolist()
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/data -v`
Expected: imports fail because data modules are absent.

- [ ] **Step 3: Implement cleaning, hash split, tokenizer training, atomic uint16 artifacts, checksums, and manifest validation**

Use NFC normalization, BLAKE2b split assignment, BPE with `<unk>`, `<bos>`, `<eos>`, `<pad>`, and a document-ending `<eos>`. Write to sibling `.tmp` files, flush, checksum, then replace final paths. `prepare_corpus` accepts an injectable iterable for tests and streams Hugging Face data in production.

- [ ] **Step 4: Verify GREEN and tiny integration**

Run: `uv run pytest tests/data -v`
Expected: all data tests pass, including corruption rejection and overwrite protection.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm/data tests/data
git commit -m "feat: prepare deterministic token corpus"
```

### Task 3: Memory-Mapped Causal Batches

**Files:**
- Create: `src/tinyllm/data/loader.py`
- Test: `tests/data/test_loader.py`

**Interfaces:**
- Consumes: verified `CorpusManifest`.
- Produces: `TokenDataset(manifest: CorpusManifest, split: str, sequence_length: int, seed: int)`, `TokenDataset.batch(indices: Tensor) -> tuple[Tensor, Tensor]`.

- [ ] **Step 1: Write failing shifted-target and reproducibility tests**

```python
def test_batch_targets_are_inputs_shifted_one_token(dataset):
    x, y = dataset.batch(torch.tensor([0, 3]))
    assert torch.equal(x[:, 1:], y[:, :-1])

def test_seed_reproduces_sample_positions(dataset_factory):
    a = dataset_factory(seed=7).sample_positions(8)
    b = dataset_factory(seed=7).sample_positions(8)
    assert torch.equal(a, b)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/data/test_loader.py -v`
Expected: `TokenDataset` import fails.

- [ ] **Step 3: Implement mmap-backed bounds-safe sampling and pinned-memory DataLoader factory**

Reject corpora shorter than `sequence_length + 1`; never sample past final token. Keep RNG generator state serializable for deterministic checkpoint resume.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/data/test_loader.py -v`
Expected: all loader tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm/data/loader.py tests/data/test_loader.py
git commit -m "feat: add causal token batches"
```

### Task 4: Transformer Primitives and GQA

**Files:**
- Create: `src/tinyllm/model/norm.py`
- Create: `src/tinyllm/model/rope.py`
- Create: `src/tinyllm/model/mlp.py`
- Create: `src/tinyllm/model/attention.py`
- Test: `tests/model/test_primitives.py`
- Test: `tests/model/test_attention.py`

**Interfaces:**
- Consumes: `ModelConfig`.
- Produces: `RMSNorm`, `RotaryEmbedding`, `SwiGLU`, `GroupedQueryAttention.forward(x: Tensor, positions: Tensor, cache: KVCache | None = None) -> tuple[Tensor, KVCache | None]`.

- [ ] **Step 1: Write failing numeric and causality tests**

```python
def test_rms_norm_has_unit_rms():
    y = RMSNorm(4)(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.allclose(y.square().mean(-1), torch.ones(1), atol=1e-5)

def test_attention_cannot_see_future_tokens(tiny_attention):
    x = torch.randn(1, 5, 16)
    changed = x.clone(); changed[:, 4] += 100
    a, _ = tiny_attention(x, torch.arange(5))
    b, _ = tiny_attention(changed, torch.arange(5))
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/model/test_primitives.py tests/model/test_attention.py -v`
Expected: model modules are missing.

- [ ] **Step 3: Implement primitives and SDPA GQA with explicit tensor shape comments**

Project Q/K/V separately, reshape Q to 8 heads and K/V to 4 heads, expand K/V groups without materialized copies where supported, apply RoPE to Q/K, and call causal SDPA.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/model/test_primitives.py tests/model/test_attention.py -v`
Expected: numeric, gradient, shape, and causality tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm/model tests/model
git commit -m "feat: implement Transformer primitives"
```

### Task 5: Decoder-Only Language Model

**Files:**
- Create: `src/tinyllm/model/block.py`
- Create: `src/tinyllm/model/transformer.py`
- Create: `src/tinyllm/model/init.py`
- Test: `tests/model/test_transformer.py`

**Interfaces:**
- Produces: `TinyLLM(config: ModelConfig)`, `TinyLLM.forward(input_ids: Tensor, targets: Tensor | None = None, caches: list[KVCache] | None = None) -> ModelOutput`, `count_parameters(model: Module) -> tuple[int, int]`.

- [ ] **Step 1: Write failing model-contract tests**

```python
def test_language_model_returns_loss_and_logits(tiny_model):
    ids = torch.randint(0, 64, (2, 8))
    out = tiny_model(ids, targets=ids)
    assert out.logits.shape == (2, 8, 64)
    assert out.loss.ndim == 0

def test_embedding_and_output_weights_are_tied(tiny_model):
    assert tiny_model.lm_head.weight is tiny_model.token_embedding.weight
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/model/test_transformer.py -v`
Expected: `TinyLLM` is missing.

- [ ] **Step 3: Implement pre-norm blocks, depth-scaled initialization, tied head, causal loss, and parameter report**

Use cross entropy with logits flattened only at loss boundary. Return a dataclass with logits, optional loss, and optional updated caches.

- [ ] **Step 4: Verify GREEN and backward pass**

Run: `uv run pytest tests/model -v`
Expected: full model suite passes and every trainable parameter receives a finite gradient in tiny backward test.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm/model tests/model
git commit -m "feat: assemble TinyLLM language model"
```

### Task 6: Optimizer, Schedule, Precision, and Training Loop

**Files:**
- Create: `src/tinyllm/training/optim.py`
- Create: `src/tinyllm/training/precision.py`
- Create: `src/tinyllm/training/metrics.py`
- Create: `src/tinyllm/training/trainer.py`
- Test: `tests/training/test_optim.py`
- Test: `tests/training/test_trainer.py`
- Test: `tests/training/test_precision.py`

**Interfaces:**
- Consumes: `TinyLLMConfig`, `TinyLLM`, `TokenDataset`.
- Produces: `build_optimizer`, `cosine_lr(step, warmup_steps, total_steps, max_lr, min_lr)`, `PrecisionPolicy.create(name, device)`, `Trainer.train(max_steps: int | None = None) -> TrainingSummary`.

- [ ] **Step 1: Write failing schedule, precision, and loss-reduction tests**

```python
def test_cosine_schedule_hits_boundaries():
    assert cosine_lr(0, 10, 100, 1e-3, 1e-4) == 0.0
    assert cosine_lr(10, 10, 100, 1e-3, 1e-4) == pytest.approx(1e-3)
    assert cosine_lr(100, 10, 100, 1e-3, 1e-4) == pytest.approx(1e-4)

def test_tiny_training_reduces_loss(tiny_training_fixture):
    summary = tiny_training_fixture.train(max_steps=30)
    assert summary.final_loss < summary.initial_loss
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/training -v`
Expected: training modules are absent.

- [ ] **Step 3: Implement AdamW grouping, accumulation, clipping, validation, JSONL/TensorBoard metrics, and finite-value guards**

Exclude biases and normalization weights from weight decay. Normalize accumulated loss by accumulation steps. Synchronize CUDA only for measured intervals. FP16 owns a GradScaler; BF16/FP32 do not.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/training -v`
Expected: CPU tiny training lowers loss; unsupported FP8 emits actionable error when TorchAO/CUDA is unavailable.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm/training tests/training
git commit -m "feat: train TinyLLM with mixed precision"
```

### Task 7: Atomic Checkpoints and Deterministic Resume

**Files:**
- Create: `src/tinyllm/training/checkpoint.py`
- Modify: `src/tinyllm/training/trainer.py`
- Test: `tests/training/test_checkpoint.py`

**Interfaces:**
- Produces: `save_checkpoint(path: Path, state: TrainingState) -> None`, `load_checkpoint(path: Path, expected: ArtifactIdentity) -> TrainingState`.

- [ ] **Step 1: Write failing round-trip and incompatible-identity tests**

```python
def test_resume_matches_uninterrupted_training(resume_fixture):
    uninterrupted = resume_fixture.run(steps=12)
    resumed = resume_fixture.run_then_resume(before=5, after=7)
    assert_state_dict_equal(uninterrupted.model, resumed.model)
    assert uninterrupted.losses == pytest.approx(resumed.losses)

def test_checkpoint_rejects_wrong_tokenizer(checkpoint_fixture):
    with pytest.raises(ValueError, match="tokenizer"):
        checkpoint_fixture.load(tokenizer_sha256="wrong")
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/training/test_checkpoint.py -v`
Expected: checkpoint API is missing.

- [ ] **Step 3: Implement atomic save/load including optimizer, scheduler, RNG, loader RNG, identities, and config snapshot**

Write temporary file, flush, then replace. Restore states before next batch sampling. Install SIGINT handling that requests save at safe step boundary.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/training/test_checkpoint.py -v`
Expected: resume trajectory matches uninterrupted run.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm/training tests/training
git commit -m "feat: resume training deterministically"
```

### Task 8: KV Cache and Text Generation

**Files:**
- Create: `src/tinyllm/model/cache.py`
- Create: `src/tinyllm/inference/sampling.py`
- Create: `src/tinyllm/inference/generate.py`
- Modify: `src/tinyllm/model/attention.py`
- Modify: `src/tinyllm/model/transformer.py`
- Test: `tests/inference/test_sampling.py`
- Test: `tests/inference/test_generation.py`

**Interfaces:**
- Produces: `KVCache`, `sample_next(logits, temperature, top_k, top_p, generator) -> Tensor`, `generate(model, input_ids, config, use_cache=True) -> Tensor`.

- [ ] **Step 1: Write failing sampling and cache-equivalence tests**

```python
def test_greedy_cached_generation_matches_uncached(tiny_model):
    prompt = torch.tensor([[1, 4, 7]])
    cached = generate(tiny_model, prompt, greedy_config(8), use_cache=True)
    plain = generate(tiny_model, prompt, greedy_config(8), use_cache=False)
    assert torch.equal(cached, plain)

def test_sampling_is_seeded():
    logits = torch.tensor([[0.1, 0.2, 0.3]])
    assert torch.equal(sample_with_seed(logits, 9), sample_with_seed(logits, 9))
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/inference -v`
Expected: inference APIs are missing.

- [ ] **Step 3: Implement layer KV append, positional offsets, greedy and temperature/top-k/top-p sampling, EOS stop**

Reject prompts longer than context. During cached decoding pass only newest token after cache priming. Keep deterministic greedy equivalence.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/inference -v`
Expected: sampling filters and cached equivalence pass.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm/model src/tinyllm/inference tests/inference
git commit -m "feat: generate text with KV cache"
```

### Task 9: FP8 Benchmark and Inference Quantization

**Files:**
- Create: `src/tinyllm/training/benchmark.py`
- Create: `src/tinyllm/inference/quantize.py`
- Test: `tests/training/test_benchmark.py`
- Test: `tests/inference/test_quantize.py`
- Test: `tests/cuda/test_precision_smoke.py`

**Interfaces:**
- Produces: `benchmark_precision(config, modes: list[str]) -> list[BenchmarkResult]`, `quantize_checkpoint(source: Path, output: Path, recipe: str) -> QuantizedManifest`.

- [ ] **Step 1: Write failing capability and metadata tests**

```python
def test_fp8_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(capabilities, "supports_fp8", lambda: False)
    with pytest.raises(RuntimeError, match="FP8.*TorchAO.*CUDA"):
        PrecisionPolicy.create("fp8", torch.device("cuda"))

def test_quantized_manifest_records_source_and_recipe(tmp_path):
    manifest = fake_quantize(tmp_path, recipe="int8-weight-only")
    assert manifest.recipe == "int8-weight-only"
    assert manifest.source_checkpoint_sha256
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/training/test_benchmark.py tests/inference/test_quantize.py -v`
Expected: benchmark and quantization modules are missing.

- [ ] **Step 3: Implement TorchAO FP8 linear conversion, synchronized benchmark, optional INT8/INT4 export, and machine-readable results**

Benchmark identical seeds and shapes after warmup, report tokens/s, peak allocated/reserved VRAM, mean loss, and finite-loss status. Quantized export validates supported recipes and never overwrites without explicit permission.

- [ ] **Step 4: Verify GREEN and optional RTX 5080 smoke tests**

Run: `uv run pytest -v -m "not cuda"`
Expected: CPU suite passes.

Run on RTX 5080: `uv run pytest tests/cuda/test_precision_smoke.py -v -m cuda`
Expected: BF16 passes; FP8 passes or skips with exact missing capability.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm tests
git commit -m "feat: benchmark FP8 and quantize inference"
```

### Task 10: Complete CLI, End-to-End Workflow, and Documentation

**Files:**
- Modify: `src/tinyllm/cli.py`
- Modify: `README.md`
- Create: `docs/architecture.md`
- Create: `docs/training-rtx-5080.md`
- Create: `tests/integration/test_tiny_workflow.py`

**Interfaces:**
- Consumes all earlier package APIs.
- Produces working `prepare`, `inspect-data`, `train`, `evaluate`, `generate`, `benchmark-precision`, and `quantize` commands.

- [ ] **Step 1: Write failing end-to-end CLI test**

```python
def test_tiny_corpus_can_prepare_train_resume_and_generate(cli_runner, tiny_text_source):
    prepared = cli_runner.invoke(["prepare", "--config", tiny_text_source.config])
    assert prepared.exit_code == 0
    trained = cli_runner.invoke(["train", "--config", tiny_text_source.config, "--max-steps", "12"])
    assert trained.exit_code == 0
    generated = cli_runner.invoke(["generate", "--checkpoint", tiny_text_source.checkpoint, "--prompt", "Once"])
    assert generated.exit_code == 0
    assert len(generated.stdout.strip()) > len("Once")
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/integration/test_tiny_workflow.py -v`
Expected: CLI commands are not fully registered.

- [ ] **Step 3: Wire commands and write reproducible documentation**

README includes quick start, exact uv commands, architecture summary, dataset/license links, disk expectations, RTX 5080 defaults, resume, generation, BF16/FP8 benchmark, INT8/INT4 export, limitations, sample metric table, and attribution. `docs/architecture.md` explains tensor shapes and KV cache. `docs/training-rtx-5080.md` records how to run and interpret local benchmark results without inventing numbers.

- [ ] **Step 4: Run full verification**

Run: `uv run ruff check .`
Expected: no lint violations.

Run: `uv run pytest -v -m "not cuda"`
Expected: all CPU tests pass.

Run: `uv run tinyllm --help`
Expected: seven commands listed.

Run: `uv run tinyllm train --config configs/tinyllm.yaml --dry-run`
Expected: validated dataset identity, ~30M parameter report, effective batch, BF16 mode, and RTX 5080 device printed without starting training.

- [ ] **Step 5: Commit**

```bash
git add src/tinyllm/cli.py README.md docs tests/integration
git commit -m "docs: complete TinyLLM workflow"
```

### Task 11: Final RTX 5080 Validation

**Files:**
- Modify after measured run: `docs/training-rtx-5080.md`
- Create from benchmark command and keep ignored: `runs/precision-benchmark.json`

**Interfaces:**
- Validates full system; adds no new production API.

- [ ] **Step 1: Prepare real corpus**

Run: `uv run tinyllm prepare --config configs/tinyllm.yaml`
Expected: checksummed train/validation binaries and tokenizer manifest created; rerun reuses them.

- [ ] **Step 2: Benchmark BF16 and FP8**

Run: `uv run tinyllm benchmark-precision --config configs/tinyllm.yaml --modes bf16,fp8 --steps 100 --warmup-steps 20`
Expected: JSON contains mode, tokens/s, peak VRAM, loss statistics, and compatibility status.

- [ ] **Step 3: Run training smoke and resume**

Run: `uv run tinyllm train --config configs/tinyllm.yaml --max-steps 200`
Expected: finite decreasing loss, metrics, and checkpoint.

Run: `uv run tinyllm train --config configs/tinyllm.yaml --resume latest --max-steps 210`
Expected: resumes at step 200 and reaches step 210.

- [ ] **Step 4: Generate and document measured results**

Run: `uv run tinyllm generate --checkpoint latest --prompt "Machine learning is" --max-new-tokens 80 --seed 42`
Expected: deterministic output for same checkpoint and seed. Add only measured throughput, VRAM, and compatibility results to documentation.

- [ ] **Step 5: Final verification and commit**

Run: `uv run ruff check . && uv run pytest -v`
Expected: lint and all available tests pass; CUDA tests skip only with explicit reason.

```bash
git add docs/training-rtx-5080.md
git commit -m "docs: record RTX 5080 validation"
```
