"""Domain errors raised by the integer coding layer."""


class CodingError(ValueError):
    """Base class for expected frequency and range-code failures."""


class InvalidFrequencyTableError(CodingError):
    """A frequency table is empty, negative, or exceeds the protocol bound."""


class UnavailableSymbolError(CodingError):
    """A symbol is outside the alphabet or has zero frequency."""


class MalformedRangeFrameError(CodingError):
    """A finite range-code frame is truncated, corrupt, or non-canonical."""


class UnsupportedRangeVersionError(CodingError):
    """A range-code frame uses an unsupported version or coder."""


class InsufficientRangeDataError(CodingError):
    """A symbol stream cannot settle the requested number of payload bits."""
