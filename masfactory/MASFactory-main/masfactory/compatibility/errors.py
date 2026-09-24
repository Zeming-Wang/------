"""Errors raised by the compatibility import layer."""


class CompatibilityImportError(ValueError):
    """Raised when an external workflow document cannot be converted into a `Graph`."""
