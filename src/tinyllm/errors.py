"""Erreurs TinyLLM destinées à l'utilisateur et limites CLI stables."""


class TinyLLMUserError(RuntimeError):
    """Classe de base des erreurs d'environnement, d'artefact et de capacité."""


class CheckpointFormatError(TinyLLMUserError):
    """Octets ou champs de checkpoint illisibles de façon sûre."""


class CheckpointCompatibilityError(TinyLLMUserError):
    """Identité de checkpoint ou version backend incompatible avec cet environnement."""


class OptionalBackendUnavailableError(TinyLLMUserError):
    """Backend optionnel ou noyau demandé indisponible."""


class PrecisionUnavailableError(OptionalBackendUnavailableError):
    """Précision numérique demandée inexécutable dans l'environnement courant."""


class DeviceUnavailableError(TinyLLMUserError):
    """Appareil d'exécution demandé indisponible."""
