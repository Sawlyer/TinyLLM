# TinyLLM Design

## Objective

Build a portfolio-grade decoder-only Transformer in PyTorch, train it once from scratch on `codelion/fineweb-edu-100M`, and make every important architectural and training choice inspectable. The default configuration targets one NVIDIA GeForce RTX 5080 with 16 GB VRAM.

## Scope

TinyLLM includes deterministic data preparation, a project-trained BPE tokenizer, the model implementation, single-GPU pretraining, validation, checkpoint resume, text generation with a KV cache, precision benchmarking, inference quantization, automated tests, and user documentation.

TinyLLM does not include distributed training, instruction tuning, RLHF, a web interface, hosted inference, or compatibility with arbitrary Hugging Face model architectures.

## Technology and Runtime

- Python 3.12
- PyTorch with CUDA
- Hugging Face `datasets` for source acquisition
- Hugging Face `tokenizers` for project-trained BPE tokenization
- Pydantic for validated configuration
- YAML configuration files
- pytest for automated tests
- TensorBoard plus JSONL metrics
- optional TorchAO for FP8 training and INT8/INT4 inference quantization
- `uv` for environment and command management

Windows is supported. Performance documentation may recommend WSL2 when measured throughput is materially better, but native Windows remains a supported path.

## Repository Structure

```text
configs/                 Default and benchmark YAML configurations
docs/                    Architecture, training, and result documentation
scripts/                 Thin command entry points
src/tinyllm/config/      Typed configuration loading and validation
src/tinyllm/data/        Download, cleaning, splitting, tokenization, binary corpus
src/tinyllm/model/       RoPE, normalization, attention, blocks, Transformer, KV cache
src/tinyllm/training/    Precision, optimizer, scheduler, checkpoints, metrics, trainer
src/tinyllm/inference/   Sampling, generation, quantization, checkpoint loading
tests/                   Unit and small integration tests
```

Each module has one responsibility. Scripts parse arguments and call package APIs; business logic remains importable and testable.

## Dataset Pipeline

The source is `codelion/fineweb-edu-100M`. Raw and processed datasets live under a configurable cache directory and are excluded from Git.

Preparation performs minimal deterministic processing:

1. Download through Hugging Face `datasets`, with cache reuse and resumability.
2. Reject missing or empty text.
3. Normalize text to Unicode NFC without rewriting content.
4. Assign documents deterministically to train or validation from a fixed seed.
5. Train a 16,384-token BPE tokenizer only on the training split.
6. Encode documents while preserving end-of-document boundaries.
7. Write memory-mapped token arrays and metadata for each split.

Metadata records source revision when available, configuration, seed, tokenizer checksum, dtype, document counts, token counts, and output checksums. Training refuses incompatible or incomplete artifacts.

The data loader samples causal sequences of length 512, creates shifted targets, supports deterministic shuffling, and uses pinned memory and worker prefetch where supported.

## Model Architecture

Default model:

- decoder-only causal Transformer
- vocabulary size: 16,384
- context length: 512
- model width: 512
- layers: 8
- query heads: 8
- key/value heads: 4
- grouped-query attention
- rotary positional embeddings
- pre-normalization with RMSNorm
- SwiGLU feed-forward network
- configurable dropout
- tied token embedding and language-model output weights
- PyTorch scaled dot-product attention

All shapes and divisibility relationships are configuration-validated. Parameter count is calculated and printed before training. Residual projections use depth-aware initialization.

Attention exposes a clear educational implementation boundary while using SDPA for the main training path. Generation uses the same weights with a layer-wise KV cache and supports context rollover within the configured maximum sequence length.

## Training

Training targets one RTX 5080. Defaults use BF16 autocast, AdamW, gradient clipping, linear warmup, cosine learning-rate decay, gradient accumulation, and optional `torch.compile`.

The requested global token batch is converted into a valid microbatch and accumulation count. Startup prints the effective batch, parameter count, trainable parameter count, expected steps, and precision mode.

Validation reports cross-entropy and perplexity on deterministic batches. Runtime metrics include step, tokens processed, learning rate, loss, validation loss, perplexity, tokens per second, elapsed time, peak VRAM, and estimated model FLOP utilization when enough device information is available.

Checkpoints contain model state, optimizer state, scheduler state, precision state, step, token count, configuration snapshot, dataset/tokenizer identity, and Python/PyTorch/CUDA RNG state. Resume rejects incompatible model, tokenizer, or corpus identities. Interrupt handling writes a recoverable checkpoint before exit when safe.

NaN or Inf loss, gradients, or parameters stops training with a diagnostic checkpoint and a concise error.

## Numeric Precision and Quantization

Training precision is independent from inference quantization.

Supported training modes:

- `bf16`: stable default for RTX 5080
- `fp16`: compatibility mode with gradient scaling
- `fp32`: reference and debugging mode
- `fp8`: experimental TorchAO conversion for eligible linear layers, requiring supported CUDA hardware, TorchAO, and `torch.compile`

FP8 does not silently fall back. Unsupported configurations fail with remediation. Optimizer states and numerically sensitive reductions remain in higher precision. FP8 and BF16 are compared using a benchmark that records throughput, peak VRAM, loss behavior, and generated output from identical seeds. BF16 remains recommended unless the local benchmark demonstrates a useful stable gain.

Inference export supports BF16 and FP16 directly, plus optional TorchAO INT8 and INT4 weight quantization where kernels support the selected device. Quantized export records its recipe and source checkpoint. Training-time FP8 and inference INT8/INT4 are presented as distinct features.

## Generation and Evaluation

Generation supports greedy decoding and sampling with temperature, top-k, top-p, a reproducible seed, maximum new tokens, and EOS termination. KV-cached output must match uncached output for deterministic decoding within numeric tolerance.

Core evaluation uses validation loss and perplexity. A small fixed prompt suite gives qualitative before/after samples without claiming general benchmark competence. Precision benchmarks use warmup iterations, synchronized timing, identical batch shapes, and machine-readable output.

## Configuration and Commands

One validated YAML hierarchy covers paths, tokenizer, model, training, logging, checkpointing, generation, precision, and quantization. Command-line overrides are supported for common experiment changes without duplicating YAML files.

Planned commands:

```text
tinyllm prepare
tinyllm inspect-data
tinyllm train
tinyllm evaluate
tinyllm generate
tinyllm benchmark-precision
tinyllm quantize
```

Every command supports `--help`. Destructive overwrite requires an explicit flag.

## Error Handling

Expected user errors produce concise messages: missing CUDA, unsupported precision, missing optional dependency, invalid configuration, corrupt artifacts, insufficient sequence length, incompatible checkpoint, and output already existing.

Partial preparation writes temporary artifacts and promotes them only after checksum validation. Existing valid cache artifacts are reused.

## Testing

Tests run without the full dataset and without requiring CUDA unless marked accordingly.

Coverage includes:

- configuration validation and overrides
- deterministic split and Unicode cleaning
- tokenizer metadata and binary corpus round trips
- causal batch construction
- RoPE, RMSNorm, GQA, SwiGLU, and shape invariants
- causality and absence of future-token leakage
- tied weights and parameter counting
- forward/backward on a tiny model
- optimizer and scheduler behavior
- checkpoint round trip and deterministic resume
- cached versus uncached generation equivalence
- sampling filters and reproducibility
- precision capability checks and explicit failures
- quantization metadata and optional CUDA smoke tests
- end-to-end tiny-corpus preparation, training, evaluation, and generation

Feature implementation follows red-green-refactor. GPU integration tests are separately marked so CPU CI remains useful.

## Documentation and Success Criteria

README explains architecture, setup, data preparation, training, resume, generation, quantization, expected disk use, and RTX 5080 configuration. It clearly distinguishes implemented-from-scratch components from PyTorch/TorchAO primitives.

Project succeeds when:

- clean setup and CPU test suite pass from documented commands;
- tiny integration training reduces loss and resumes deterministically;
- full preparation produces verified train/validation artifacts;
- RTX 5080 BF16 training runs without memory errors using default configuration;
- generated text and validation metrics can be reproduced from a checkpoint;
- KV cache matches uncached deterministic generation;
- FP8 benchmark either records a stable result or reports precise incompatibility;
- inference quantization is optional, configurable, and never confused with training precision.

## Delivery Order

Implementation proceeds through independently testable slices: project foundation and configuration; dataset/tokenizer pipeline; model primitives and full Transformer; training infrastructure; checkpointing and metrics; generation and KV cache; precision benchmarking and quantization; end-to-end documentation and verification.
