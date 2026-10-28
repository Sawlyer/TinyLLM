# TinyLLM

TinyLLM is my compact, decoder-only language-model project: data preparation, tokenizer
training, pre-training, deterministic resume, evaluation, cached generation, precision
benchmarking, and inference export in one readable PyTorch codebase. I built it to make every
stage inspectable, from corpus checksums to grouped-query attention and atomic checkpoints.

Current default configuration contains 39,854,592 trainable parameters. `tinyllm train
--dry-run` computes this value from configuration instead of relying on a stale estimate.

```mermaid
flowchart LR
    source["Text source"] --> prepare["prepare"]
    prepare --> corpus["Checksummed corpus and tokenizer"]
    corpus --> train["train or resume"]
    train --> checkpoint["Checkpoint"]
    train --> metrics["JSONL, CSV, TensorBoard"]
    train --> manifest["Run manifest"]
    checkpoint --> evaluate["evaluate"]
    checkpoint --> generate["generate"]
    checkpoint --> export["quantize and validate"]
```

## What is included

- streaming preparation from `codelion/fineweb-edu-100M`, with deterministic train/validation
  split, locally trained BPE tokenizer, SHA-256 identities, and atomic artifact publication;
- pre-norm Transformer blocks with RMSNorm, RoPE, grouped-query attention, SwiGLU, tied input
  and output embeddings, and one KV cache per layer;
- single-device training with FP32, FP16, BF16, or optional FP8, gradient accumulation,
  validation metrics in JSONL/CSV and TensorBoard, atomic checkpoints, and deterministic resume;
- greedy or sampled generation with rolling context, precision comparison, and reload-verified
  BF16, FP16, INT8, or INT4 inference exports;
- CPU test suite plus separately marked CUDA smoke tests.

Architecture details live in [docs/architecture.md](docs/architecture.md). RTX 5080 setup and
benchmark procedure live in [docs/training-rtx-5080.md](docs/training-rtx-5080.md).

## Windows quick start with uv

Requirements: Windows 10 or newer, Git, and Python 3.12. `uv` publishes an official Windows
installer and Windows x86-64 binaries: [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```powershell
winget install --id=astral-sh.uv -e
git clone <repository-url> TinyLLM
cd TinyLLM
uv sync --extra dev
uv run pytest -m "not cuda"
uv run tinyllm --help
```

`uv sync` is enough for CPU development and tests. CUDA wheels depend on current PyTorch and
driver compatibility; select Windows, Pip, Python, and matching CUDA on the official
[PyTorch installer](https://pytorch.org/get-started/locally/), then run its generated command
inside this environment. Confirm actual runtime before training:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

TorchAO is optional. Install project extra only when testing FP8 or INT8/INT4 paths:

```powershell
uv sync --extra dev --extra fp8
```

Installation alone does not guarantee kernel support. FP8 command probes requested CUDA device
and real forward/backward kernel; unsupported combinations fail with reason instead of silently
falling back. TorchAO documents current
[quantized training](https://docs.pytorch.org/ao/stable/workflows/training.html) and
[quantized inference](https://docs.pytorch.org/ao/stable/workflows/inference.html) support.

## Prepare and inspect data

Default source is [`codelion/fineweb-edu-100M`](https://huggingface.co/datasets/codelion/fineweb-edu-100M),
a 100M-token reservoir sample of
[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
Preparation resolves remote revision, records upstream fingerprint, trains tokenizer, writes
`uint16` token artifacts, and validates every checksum before publication.

```powershell
uv run tinyllm prepare --config configs/tinyllm.yaml
uv run tinyllm inspect-data --config configs/tinyllm.yaml
uv run tinyllm audit-artifacts --manifest data/processed/manifest.json
uv run tinyllm diagnose-tokenizer --manifest data/processed/manifest.json --splits train validation --sample-documents 1000 --compare-vocab-sizes 8192 16384 --output outputs/tokenizer-diagnostics.json
```

`audit-artifacts` reads a manifest and its referenced tokenizer and token files without changing
them. It works offline without loading a model or initializing CUDA, prints a JSON report, and
returns a failing exit code when it finds invalid metadata, missing or truncated files, checksum
mismatches, tokenizer identity changes, or token IDs outside the vocabulary. Legacy manifests
remain untouched and produce a compatibility warning.
`diagnose-tokenizer` validates the manifest, tokenizer and token artifact checksums before reading.
Token totals, vocabulary use and document-length percentiles cover each selected split. Quality
signals and decoded examples use the first `--sample-documents` documents, in stable corpus order.
Optional vocabulary comparisons retrain BPE configurations on the same decoded sample from the
train artifact; results describe that bounded sample, not the original unprocessed source.
The command never rewrites corpus artifacts. Its JSON report prints to stdout unless `--output`
is set; existing reports require `--overwrite`.

Outputs never overwrite by default. Rebuilding intentional dataset requires explicit
`--overwrite`. For offline smoke run, set `data.dataset_name` to local UTF-8 text file; each
non-empty line becomes one document.

Dataset card for 100M sample does not declare separate license. Upstream FineWeb-Edu card
declares [Open Data Commons Attribution 1.0](https://opendatacommons.org/licenses/by/1-0/).
Web documents can retain their own terms and sensitive or inaccurate content; inspect source
card and intended use before redistributing artifacts or outputs.

### Disk planning

Dataset card currently reports 329 MB Parquet download. One 100M-token `uint16` corpus is about
200 MB before filesystem overhead. Download cache, staging files, tokenizer, logs, and prepared
corpus can coexist, so reserve at least 1 GB for data work. Default model checkpoint after AdamW
state initialization is expected to occupy several hundred MB; multiple checkpoints and exports
make several GB prudent. These are capacity estimates from file formats and parameter count, not
measured throughput or memory results.

## Train, resume, evaluate, generate

Validate configuration without allocating CUDA training state or starting optimizer steps:

```powershell
uv run tinyllm train --config configs/tinyllm.yaml --dry-run
```

Dry run emits a JSON preflight report. It checks prepared corpus compatibility when artifacts exist,
device and precision support, output paths, checkpoint collisions, and estimated checkpoint storage.
It does not download data, create output files, instantiate the model, or allocate CUDA tensors.
Missing prepared data and deferred FP8 kernel probing appear as warnings; predictable failures appear
with a field, path, cause, and suggested fix. Default effective batch is 16 micro sequences times 8
accumulation steps: 128 sequences or 65,536 tokens per optimizer step.

```powershell
uv run tinyllm train --config configs/tinyllm.yaml --max-steps 20 --checkpoint checkpoints/latest.pt
uv run tinyllm train --config configs/tinyllm.yaml --resume checkpoints/latest.pt --checkpoint checkpoints/resumed.pt --max-steps 10000
uv run tinyllm evaluate --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --batches 8
uv run tinyllm generate --checkpoint checkpoints/resumed.pt --prompt "Once upon a time"
uv run tinyllm generate --checkpoint checkpoints/resumed.pt --prompt "A fox" --prompt "A lake" --temperature 0.8 --top-k 40 --top-p 0.9 --repetition-penalty 1.1 --stop "<eos>" --json
uv run tinyllm interactive --checkpoint checkpoints/resumed.pt
uv run tinyllm interactive --checkpoint checkpoints/resumed.pt --history-file outputs/session.jsonl
uv run tinyllm profile-training --config configs/tinyllm.yaml --warmup-steps 2 --steps 3 --trace outputs/profile.json --output outputs/report.json
```

Checkpoint stores model, optimizer, schedule boundary, precision state, Python/PyTorch/CUDA RNG,
loader RNG, full configuration snapshot, and model/tokenizer/corpus identities. Resume rejects
incompatible configuration or artifacts before mutation. Generation can recover configuration
from checkpoint; pass `--config` when intentionally validating against external configuration.
Repeat `--prompt` to generate several prompts sequentially; each uses the configured seed. Sampling
applies repetition penalty first, then temperature, top-k, then top-p. Repetition penalty defaults
to `1.0` (disabled); positive logits are divided by it and negative logits multiplied by it. Repeat
`--stop` for text sequences that end generation; stop tokens are omitted from the returned text.
`--json` prints effective parameters plus per-prompt generated token count, tokens/second, and
first-token latency in milliseconds. Timing starts after model loading and prompt tokenization.
`interactive` loads a checkpoint once, then accepts independent prompts. Use `/set temperature 0.8`,
`/set top_k 40`, `/history`, `/reset`, and `/quit`. Press Ctrl+C during generation to stop that
response; press it at the prompt to leave. Prompt and response text stay in memory unless
`--history-file` is explicitly supplied. This mode does not add prior turns to later prompts: a
chat-style format only works when the model's training data taught it that format.
`profile-training` runs a short profile against prepared training data. One priming step and the
configured warmup steps are excluded from measured summaries; model/dataset setup is timed
separately. The report includes
tokens/second, batch loading and transfer time, model/optimizer time, validation time, allocated and
reserved CUDA memory, device and library versions, and total profiling duration. PyTorch operator scopes
separate forward, backward, optimizer, data loading, and validation. `--trace` exports an optional
Chrome trace. Profile updates stay in memory; logs and checkpoints use a temporary directory and
are discarded when the command exits. CUDA transfer timing synchronizes before and after each
profiled batch, adding overhead for clearer attribution. Use `--output` to keep the JSON report.
Default configuration keeps `training.max_steps: 10000`; first command stops early through CLI
without changing checkpoint configuration snapshot. Resume therefore remains configuration-compatible
and continues toward step 10000.

Training checkpoints use a versioned format and retain producer and UTC creation metadata. Legacy
v2/v3 files migrate in memory when read; unsupported future formats return an upgrade instruction.
Inference exports use a separate weights-only artifact without optimizer or RNG state:

```powershell
uv run tinyllm export-weights --checkpoint checkpoints/latest.pt --output exports/tinyllm-weights.pt
uv run tinyllm generate --checkpoint exports/tinyllm-weights.pt --prompt "Once upon a time"
```

Training resume rejects inference-only weights. Both artifact types remain readable by `generate`
and `evaluate`; existing training checkpoints remain valid inference inputs.

`Ctrl+C` and `SIGTERM` finish the current optimizer step, write a resumable `interrupt.pt` in the
checkpoint directory, then stop. Abrupt termination that cannot deliver a signal can only resume
from the latest periodic `step-*.pt` checkpoint.

Inspect saved checkpoints before cleaning a directory. `list` only reads file names, sizes, and
timestamps; `inspect` validates and deserializes one checkpoint on CPU without constructing a model:
Inspection et plan de rétention chargent chaque checkpoint vérifié en mémoire CPU; prévoir une RAM
au moins comparable à la taille du plus gros fichier.

```powershell
uv run tinyllm checkpoints list --directory checkpoints
uv run tinyllm checkpoints inspect --checkpoint checkpoints/step-00001000.pt
```

Retention defaults to a preview. It keeps the latest valid resumable checkpoint, the best checkpoint
with a matching `validation_step` in the selected metrics file, and the configured number of newest
periodic checkpoints. `--keep-every` additionally keeps periodic steps divisible by that interval.
Named files such as `latest.pt`, `interrupt.pt`, and `diagnostic.pt` remain untouched. Use
`--apply` only after reviewing the printed deletion plan. If metrics contain multiple runs, pass
`--run-id`; no files are removed when no valid resumable checkpoint or matching best metric exists.

```powershell
uv run tinyllm checkpoints prune --directory checkpoints --metrics runs/metrics.jsonl --keep-last 2 --keep-every 5000
uv run tinyllm checkpoints prune --directory checkpoints --metrics runs/metrics.jsonl --keep-last 2 --keep-every 5000 --apply
```

Checkpoint writes use a temporary file and atomic replacement, so a failed save leaves an existing
checkpoint at its destination intact. Pruning is never automatic.

### Training metrics

Each training report appends one record to `runs/metrics.jsonl` and one row to
`runs/metrics.csv`. TensorBoard remains available in `runs/tensorboard`. JSONL and CSV flush after
each report. A CSV schema change adds columns while preserving earlier rows.

`timestamp_utc` is the record time in ISO 8601 UTC. `run_id` matches `runs/run.json`; `session_id`
identifies one training process. Resuming keeps the run ID and starts a new session, so repeated
step numbers remain distinguishable. Steps increase within each session.

- `loss`, `validation_loss`: mean training loss and latest validation loss, unitless. `validation_step`
  gives the step used for that validation value. Perplexity fields are exponentiated losses.
- `learning_rate`: optimizer learning rate at the reported step. `gradient_norm`: gradient norm
  before clipping.
- `tokens_processed`: cumulative tokens in the run. `tokens_per_second` and `sequences_per_second`
  cover the training interval since the previous report; validation and metric writes are excluded.
- `data_loading_seconds`: time spent waiting for prefetched training batches in that interval.
  `evaluation_seconds`: time spent evaluating validation data in the interval.
- `elapsed_seconds`: wall time since the current session started. `peak_vram_bytes`: peak allocated
  CUDA memory since the last reset. `cuda_free_memory_bytes` and `cuda_total_memory_bytes` record
  free and total CUDA memory at report time; CUDA fields are absent on CPU runs.

### Reproducible evaluation

Evaluation defaults to the validation split, 8 batches, and seed 42. Each batch samples token
windows with replacement from the selected corpus split. The local evaluation generator leaves
training loader state unchanged. `--tokens` requests an exact target-token count; the last window
is shortened when needed. Every target token, including EOS, contributes to loss. The model's
causal attention mask remains active.

```powershell
uv run tinyllm evaluate --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --split validation --batches 8 --seed 42
uv run tinyllm evaluate --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --split validation --tokens 100000 --seed 42 --output runs/evaluation.json
uv run tinyllm evaluate --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --split validation --tokens 100000 --seed 42 --output runs/evaluation.csv
```

Reports include checkpoint and selected corpus SHA-256, split, seed, requested and actual token
counts, batch count, mean next-token loss, population standard deviation across batch mean losses,
and perplexity. Loss and its dispersion use nats per predicted token; perplexity is the exponential
of token-weighted mean loss and has no unit. For numeric safety, values above `exp(80)` are capped;
`perplexity_saturated` marks this case. JSON and CSV output refuse to replace an existing file
unless `--overwrite` is passed. Human-readable lines go to stderr; stdout remains one JSON object
for scripts. The JSON keeps the existing `validation_loss` and `validation_perplexity` fields.

For repeat runs with the same checkpoint, corpus, split, seed, batch or token limit, device, dtype,
PyTorch version, and CUDA runtime, compare floating-point results with relative tolerance `1e-4`
and absolute tolerance `1e-6`. This tolerance does not promise equivalence across devices, dtypes,
or software versions.

One-off overrides use validated dotted YAML scalars:

```powershell
uv run tinyllm train --config configs/tinyllm.yaml --set training.max_steps=2000 --set training.warmup_steps=100 --dry-run
```

Unknown keys and invalid values fail validation. Keep `training.warmup_steps` below
`training.max_steps`.

## BF16/FP8 benchmark

Benchmark modes receive same initialized weights, seed, batch shapes, and pre-generated token
batches. Warmup is excluded; CUDA synchronization brackets measured section. Results include
tokens/s, elapsed time, peak allocated/reserved bytes, mean loss, and finite-loss flag.

```powershell
uv run tinyllm benchmark-precision --config configs/tinyllm.yaml --modes bf16 fp8 --warmup-steps 10 --measured-steps 50 --output runs/precision-benchmark.json
```

Existing output is refused unless `--overwrite` is explicit. Compare local results only after
confirming same software, configuration, thermals, and background load.

Sample reporting table, intentionally unpopulated until local hardware run:

| Mode | Tokens/s | Peak allocated | Peak reserved | Mean loss | Finite |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 | — | — | — | — | — |
| FP8 | — | — | — | — | — |

## BF16, FP16, INT8, and INT4 export

```powershell
uv run tinyllm quantize --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --recipe bf16 --output exports/tinyllm-bf16.pt
uv run tinyllm quantize --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --recipe int8 --output exports/tinyllm-int8.pt
uv run tinyllm quantize --config configs/tinyllm.yaml --checkpoint checkpoints/resumed.pt --recipe int4 --output exports/tinyllm-int4.pt
```

BF16/FP16 direct exports require supported execution dtype. INT8/INT4 are TorchAO weight-only
paths and remain conditional on compatible library version, device, serialization support, and
kernels. Export succeeds only after strict reload and deterministic numeric round trip. Existing
destination is preserved on failure and replaced only with `--overwrite`.

## Limitations

- single process and single device; no distributed training;
- 512-token default context and compact parameter budget;
- pre-training only: no instruction tuning, preference tuning, safety tuning, or factuality
  guarantee;
- dataset is predominantly English web text and can carry source bias, duplication, personal
  data, or unsafe material;
- default 100M-token sample is useful for experiments, not competitive general-purpose quality;
- FP8, INT8, and INT4 support must be established on target hardware; CPU tests cannot validate
  CUDA kernel availability or speed;
- benchmark output is local evidence, never universal hardware claim.

## Attribution

- 100M sample: [codelion/fineweb-edu-100M dataset card](https://huggingface.co/datasets/codelion/fineweb-edu-100M)
- upstream corpus: [HuggingFaceFW/fineweb-edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- FineWeb paper: [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557)
- dataset sampling context: [The 1 Billion Token Challenge](https://huggingface.co/blog/codelion/optimal-dataset-mixing/)
- PyTorch, Hugging Face Datasets, Tokenizers, TensorBoard, Pydantic, PyYAML, and optional
  TorchAO power runtime and tooling.

