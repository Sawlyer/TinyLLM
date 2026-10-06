# Final fix wave report

Date: 2026-08-21 (Europe/Paris)

Worktree: `C:\Users\Utilisateur\Desktop\Developpement\TinyLLM\.worktrees\tinyllm`

Branch: `feat/tinyllm`

Base commit: `d8fda1d`

## Outcome

All blocking final-review findings are addressed in one TDD wave. Training resume remains strict.
Evaluation, generation, and quantization now use a model-only checkpoint boundary. No push was
performed.

## RED evidence

Tests were added before production changes.

- Fresh-process TorchAO export/reload failed with `_pickle.UnpicklingError` and
  `Unsupported global: GLOBAL torchao.quantization.ReloadTensor`. Failure occurred during the
  export round-trip because `torch.load(weights_only=True)` ran before backend import and safe
  type registration.
- Prefetch tests failed at import because `DeterministicBatchPrefetcher` did not exist. Trainer
  tests also exposed direct `TokenDataset.sample_positions` use in production.
- Model-only checkpoint tests failed at import because `load_model_checkpoint` and
  `CheckpointFormatError` did not exist.
- Corpus/model compatibility tests failed at import because manifest tokenizer identity and
  validation APIs did not exist.
- Benchmark tests failed at import because `BenchmarkExecutionError` did not exist. Existing
  behavior also returned `status=ready` with `finite_loss=false`.
- Non-finite trainer tests showed only an exception, without durable diagnostic state or reason.
- CLI integration tests showed evaluate/generate constructing `Trainer`, loading training corpus,
  optimizer/RNG state, and applying training-device topology.

First combined GREEN command after implementation:

```powershell
.venv\Scripts\python.exe -m pytest tests/inference/test_quantize_subprocess.py tests/data/test_prefetcher.py tests/training/test_model_checkpoint.py tests/data/test_model_compatibility.py tests/training/test_benchmark.py tests/training/test_trainer.py tests/inference/test_quantize.py tests/integration/test_tiny_workflow.py -p no:cacheprovider --basetemp "C:\Users\Utilisateur\Documents\ChatGPT\Projet Github\.pytest-tinyllm-finalfix-run3" -q
```

Result:

```text
72 passed in 13.24s
```

## Fixes

### TorchAO INT8/INT4 safe reload

- Quantized format advanced from v2 to v3.
- Every v3 export writes a checksummed `<checkpoint>.manifest.json` preflight sidecar.
- Loader reads preflight JSON and verifies main checkpoint SHA-256 before deserialization.
- Exact PyTorch version, TorchAO version, recipe config class, config parameters, and tensor
  serialization version are validated before `torch.load`.
- Unsafe checkpoint globals are inspected without unpickling. Only exact `torchao.*` globals
  declared by manifest tensor-subclass metadata are resolved and passed through a scoped
  `torch.serialization.safe_globals` context.
- Embedded manifest must equal preflight manifest after load. Mismatch fails closed.
- Legacy v2 direct-precision checkpoints remain loadable. Legacy weight-only checkpoints with
  unsafe globals and no preflight are rejected with re-export guidance.
- Source training checkpoint uses model-only loader; optimizer and RNG state are never loaded for
  export.
- Export promotes main file and sidecar transactionally. Failure restores previous pair or leaves
  no new output.
- Fresh-process subprocess test covers both `int8-weight-only` and `int4-weight-only`, then runs
  autoregressive generation.

Real installed TorchAO check:

```text
TorchAO 0.18.0 INT8 export: success
fresh Python process reload: success
generation shape: [1, 5]
forward logits finite: true
```

### Deterministic pinned prefetch pipeline

- Trainer now consumes `DeterministicBatchPrefetcher` instead of sampling batches directly.
- Prefetch uses a private preview generator. Construction and lookahead do not advance checkpoint
  RNG.
- Dataset RNG commits only when one batch is consumed.
- Resume resets prefetch queue from saved consumed-boundary RNG, reproducing exact next batch.
- Host batches are pinned when CUDA or MPS pinning support is available. Device transfers use
  `non_blocking=True` on CUDA.
- Training checkpoint loader RNG contract remains unchanged and strict.

### Atomic diagnostics for NaN/Inf

- Loss, gradients, gradient norm, and post-step parameters are checked.
- Any non-finite value writes `diagnostic.pt` atomically before raising.
- Training checkpoint format advanced from v2 to v3 with `status` and `reason`.
- Diagnostic checkpoint contains model, optimizer, precision, loader, RNG, step, token count,
  identity, `status=diagnostic`, and exact reason.
- Diagnostic checkpoints cannot resume training or be used as inference checkpoints.
- Legacy resumable v2 checkpoints remain supported.

### Tokenizer/model/corpus identity

- Corpus manifest advanced from v1 to v2.
- Preparation records actual tokenizer vocabulary size and exact IDs for `<unk>`, `<bos>`,
  `<eos>`, and `<pad>`.
- Manifest load verifies tokenizer checksum, parses real tokenizer, and compares recorded
  vocabulary/special IDs.
- v1 manifests migrate in memory from checksummed tokenizer identity. Legacy serialization still
  omits v2-only fields.
- Training/evaluation validate tokenizer vocabulary and special IDs against
  `ModelConfig.vocab_size`.
- Selected corpus artifacts are scanned for token IDs outside model embedding range.
- Generation validates checkpoint tokenizer checksum, actual vocabulary, special IDs, prompt IDs,
  and model config before forward execution.

### Benchmark failure semantics

- Non-finite measured loss returns `status=failed`, `finite_loss=false`,
  `reason=non-finite measured loss`, and no numeric mean loss.
- Gradients are checked after backward. Parameters are checked after optimizer step.
- Declared `BenchmarkExecutionError` is recorded per mode; generic `RuntimeError` programming bugs
  propagate unchanged.
- Failed benchmark JSON is written before CLI exits 1. Unsupported optional precision remains
  separately represented.

### Narrow user-facing errors

- Added `TinyLLMUserError` hierarchy for checkpoint format, compatibility, optional backend,
  precision, and device failures.
- Training and quantized checkpoint loaders explicitly map `pickle.UnpicklingError`, archive I/O,
  and known serialization-runtime failures.
- TorchAO and FP8 optional API failures use dedicated exceptions.
- FP8 probe converts only known TorchAO/TorchDynamo/Inductor/Triton failures to capability reasons.
  Generic runtime failures propagate unchanged.
- CLI catches expected user-facing failures, not arbitrary `RuntimeError` bugs.

### Model-only inference and CLI

- `load_model_checkpoint` strictly loads model state and identity while ignoring optimizer,
  scheduler, precision scaler, RNG, loader state, and CUDA topology.
- Training resume still uses full strict checkpoint loader and validates CUDA/device topology.
- Evaluate loads validation split only. Missing train artifact no longer blocks evaluation.
- Generate needs tokenizer file only. Missing corpus and manifest no longer block generation.
- Quantized generation is available through `generate --quantized --config ...`.
- Evaluate/generate accept `--device`; explicit CPU loads a checkpoint created with CUDA training
  metadata.
- Generate accepts direct `--seed` override.
- Dry-run prints expected document-step count, immutable dataset revision/fingerprint, tokenizer
  checksum, and train checksum.

## Final verification

Non-CUDA suite:

```powershell
.venv\Scripts\python.exe -m pytest -m "not cuda" -p no:cacheprovider --basetemp "C:\Users\Utilisateur\Documents\ChatGPT\Projet Github\.pytest-tinyllm-finalfix-final-cpu" -q
```

```text
226 passed, 2 deselected in 16.23s
```

CUDA smoke:

```powershell
.venv\Scripts\python.exe -m pytest -m cuda -p no:cacheprovider --basetemp "C:\Users\Utilisateur\Documents\ChatGPT\Projet Github\.pytest-tinyllm-finalfix-final-cuda" -q -rs
```

```text
1 passed, 1 skipped, 226 deselected in 3.69s
SKIPPED: FP8 unavailable: Float8Linear compiled kernel probe failed on cuda: TritonMissing
```

Ruff:

```powershell
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check <24 changed Python files>
```

```text
All checks passed!
24 files already formatted
```

Additional checks:

```text
git diff --check: clean
fresh-process fake TorchAO INT8 reload/generate: pass
fresh-process fake TorchAO INT4 reload/generate: pass
real TorchAO 0.18.0 INT8 fresh reload/generate: pass
CLI prepare/train/evaluate/generate/quantize/quantized-generate workflow: pass
```

## Remaining concerns

- Real TorchAO INT4 execution was unavailable in installed environment because TorchAO reported
  `Requires mslk >= 1.0.0`. Failure is now explicit `OptionalBackendUnavailableError`. Deterministic
  fresh-process INT4 serialization/reload/generation is covered with fake TorchAO tensor subclass.
- FP8 CUDA smoke remains skipped because Windows environment lacks working Triton. BF16 CUDA smoke
  passes. Existing Task 11 report already contains real RTX 5080 BF16 benchmark, step-200 training,
  step-210 resume, and deterministic generation measurements.
- Task 11 historical report states direct `--seed` did not exist. That statement remains accurate
  for measured commit `d8fda1d`; current CLI now supports `--seed`.
- Weight-only v3 checkpoint and `.manifest.json` sidecar form one export. Copying only `.pt` fails
  closed. Pair promotion has no single-filesystem-operation atomicity, but checksum validation and
  rollback prevent accepting a mixed pair.
- Full 200-step training measurement was not repeated. Deterministic resume regression tests,
  end-to-end tiny workflow, CUDA BF16 smoke, and prior Task 11 measurements cover changed boundaries.

No push performed.
