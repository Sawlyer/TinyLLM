from pathlib import Path

import pytest

from tinyllm.config.load import load_config


@pytest.fixture
def config_path() -> Path:
    return Path(__file__).parents[2] / "configs" / "tinyllm.yaml"


def test_default_config_has_valid_gqa_shape(config_path: Path) -> None:
    cfg = load_config(config_path, [])

    assert cfg.model.n_heads == 8
    assert cfg.model.n_kv_heads == 4
    assert cfg.model.d_model % cfg.model.n_heads == 0


def test_override_changes_nested_value(config_path: Path) -> None:
    cfg = load_config(config_path, ["training.precision=fp32"])

    assert cfg.training.precision == "fp32"


def test_invalid_head_ratio_is_rejected(config_path: Path) -> None:
    with pytest.raises(ValueError, match="n_heads.*n_kv_heads"):
        load_config(config_path, ["model.n_kv_heads=3"])


def test_warmup_must_leave_at_least_one_decay_step(config_path: Path) -> None:
    with pytest.raises(ValueError, match="warmup_steps.*max_steps"):
        load_config(config_path, ["training.warmup_steps=10000"])


def test_override_parses_yaml_null_scalar(config_path: Path) -> None:
    cfg = load_config(config_path, ["generation.top_k=null"])

    assert cfg.generation.top_k is None


def test_override_rejects_unknown_nested_key(config_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown configuration key"):
        load_config(config_path, ["training.unknown=true"])


def test_override_rejects_yaml_collection(config_path: Path) -> None:
    with pytest.raises(ValueError, match="YAML scalars"):
        load_config(config_path, ["generation.top_k=[1, 2]"])


def test_malformed_yaml_override_is_reported_as_invalid_value(config_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid YAML override value: generation.top_k"):
        load_config(config_path, ["generation.top_k=[unterminated"])


def test_malformed_yaml_is_reported_as_invalid_configuration(tmp_path: Path) -> None:
    malformed = tmp_path / "broken.yaml"
    malformed.write_text("training: [unterminated", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML configuration"):
        load_config(malformed, [])
