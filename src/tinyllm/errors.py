"""Expected user-facing TinyLLM failures with stable CLI boundaries."""


class TinyLLMUserError(RuntimeError):
    """Base class for expected environment, artifact, and capability failures."""


class CheckpointFormatError(TinyLLMUserError):
    """Checkpoint bytes or fields are not safely readable."""


class CheckpointCompatibilityError(TinyLLMUserError):
    """Checkpoint identity or backend version cannot run in this environment."""


class OptionalBackendUnavailableError(TinyLLMUserError):
    """Requested optional backend or kernel is unavailable."""


class PrecisionUnavailableError(OptionalBackendUnavailableError):
    """Requested numeric precision cannot run in the current environment."""


class DeviceUnavailableError(TinyLLMUserError):
    """Requested execution device is unavailable."""
