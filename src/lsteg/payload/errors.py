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
