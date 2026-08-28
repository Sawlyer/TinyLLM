from pathlib import Path

import pytest

from tinyllm.cli import _build_parser
from tinyllm.config.load import load_config
from tinyllm.inference.quantize import SUPPORTED_RECIPES


def test_public_surface_excludes_unvalidated_fp8_and_int4() -> None:
    parser = _build_parser()
    exposed_choices = {
        str(choice).lower()
        for action in parser._actions
        for subparser in (getattr(action, "choices", None) or {}).values()
        for sub_action in subparser._actions
        for choice in (sub_action.choices or ())
    }

    assert "fp8" not in exposed_choices
    assert "int4" not in exposed_choices
    assert "int4-weight-only" not in SUPPORTED_RECIPES


def test_configuration_rejects_fp8_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        load_config(Path("configs/tinyllm.yaml"), ["training.precision=fp8"])
