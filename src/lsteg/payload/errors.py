"""Domain errors raised by the payload layer."""


class PayloadCodecError(ValueError):
    """Base class for expected payload validation failures."""


class InvalidSecretTextError(PayloadCodecError):
    """The supplied secret cannot be represented by the text payload protocol."""


class SecretTooLongError(InvalidSecretTextError):
    """The normalized secret exceeds the protocol code-point limit."""


class MalformedPayloadError(PayloadCodecError):
    """A payload frame is truncated, corrupt, non-canonical, or out of bounds."""


class UnsupportedPayloadVersionError(PayloadCodecError):
    """A payload frame uses an unsupported protocol version."""


class UnsupportedPayloadAlgorithmError(PayloadCodecError):
    """A payload frame requests an unsupported cryptographic algorithm."""


class AuthenticationError(PayloadCodecError):
    """A secure payload could not be authenticated with the supplied key."""


class KeyManagementError(ValueError):
    """Base class for key validation and key-file failures."""


class InvalidMasterKeyError(KeyManagementError):
    """A master key or serialized key file violates the versioned format."""


class KeyFileExistsError(KeyManagementError):
    """A key file already exists and must not be overwritten."""


class KeyFileWriteError(KeyManagementError):
    """A master key could not be installed safely on disk."""
