"""Validated configuration models for TinyLLM."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    """Base model that rejects misspelled configuration keys."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictConfig):
    dataset_name: str
    dataset_revision: str | None = None
    cache_dir: Path
    output_dir: Path
    text_field: str = "text"
    validation_fraction: float = Field(gt=0, lt=1)
    seed: int


class TokenizerConfig(StrictConfig):
    vocab_size: int = Field(gt=0, le=65536)
    min_frequency: int = Field(ge=1)
    special_tokens: list[str]
    path: Path


class ModelConfig(StrictConfig):
    vocab_size: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    d_model: int = Field(gt=0)
    n_layers: int = Field(gt=0)
    n_heads: int = Field(gt=0)
    n_kv_heads: int = Field(gt=0)
    mlp_ratio: float = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    rope_theta: float = Field(gt=0)
    rms_norm_eps: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_attention_shape(self) -> "ModelConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        return self


class TrainingConfig(StrictConfig):
    seed: int
    device: str
    precision: str
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    min_learning_rate: float = Field(ge=0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    compile: bool = False
    checkpoint_dir: Path

    @model_validator(mode="after")
    def validate_training_options(self) -> "TrainingConfig":
        allowed_precisions = {"bf16", "fp16", "fp32", "fp8"}
        if self.precision not in allowed_precisions:
            raise ValueError(f"precision must be one of {', '.join(sorted(allowed_precisions))}")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be less than max_steps")
        return self


class LoggingConfig(StrictConfig):
    run_dir: Path
    log_interval: int = Field(gt=0)
    validation_interval: int = Field(gt=0)
    checkpoint_interval: int = Field(gt=0)


class GenerationConfig(StrictConfig):
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0)
    top_k: int | None = Field(default=None, gt=0)
    top_p: float = Field(gt=0, le=1)
    seed: int
    eos_token_id: int | None = Field(default=None, ge=0)


class QuantizationConfig(StrictConfig):
    recipe: str | None = None
    output_dir: Path


class TinyLLMConfig(StrictConfig):
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    generation: GenerationConfig
    quantization: QuantizationConfig
