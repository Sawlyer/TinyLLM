import warnings

_SCRIPT_METHOD_DEPRECATION = (
    "`torch.jit.script_method` is deprecated. Please switch to `torch.compile` or "
    "`torch.export`."
)


def test_torch_script_filter_does_not_hide_tinyllm_deprecations() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.warn_explicit(
            _SCRIPT_METHOD_DEPRECATION,
            DeprecationWarning,
            filename="torch/jit/_script.py",
            lineno=365,
            module="torch.jit._script",
        )
        warnings.warn_explicit(
            _SCRIPT_METHOD_DEPRECATION,
            DeprecationWarning,
            filename="tinyllm/training/precision.py",
            lineno=1,
            module="tinyllm.training.precision",
        )

    assert [warning.filename for warning in captured] == [
        "tinyllm/training/precision.py"
    ]
