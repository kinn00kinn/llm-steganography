"""Domain errors raised by the language-model boundary."""


class ModelBackendError(RuntimeError):
    """Base class for expected model loading and inference failures."""


class InvalidModelManifestError(ValueError):
    """A model manifest is malformed, incomplete, or unsafe."""


class ModelDependencyError(ModelBackendError):
    """The pinned optional model runtime is unavailable or mismatched."""


class ModelArtifactError(ModelBackendError):
    """A pinned model/tokenizer artifact cannot be loaded or verified."""


class ModelDeviceError(ModelBackendError):
    """The manifest's required inference device is unavailable."""


class ModelInputError(ValueError):
    """Token IDs or context length violate the backend contract."""
